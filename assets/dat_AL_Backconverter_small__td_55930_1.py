import numpy as np
import colorsys
import pickle
import os
import json
from scipy import signal
from scipy.optimize import minimize
from scipy.signal import find_peaks
import socketio
import math

import sys, os

project_folder = os.path.dirname(__file__)
if project_folder not in sys.path:
    sys.path.append(project_folder)
import shared

# --- Configuration ---
server_url = 'http://localhost:5001'  # Replace with your server's address

# --- SocketIO Client Setup ---
sio = socketio.Client()

@sio.event
def connect():
    print('TouchDesigner: Connected to server!')

@sio.event
def disconnect():
    print('TouchDesigner: Disconnected from server!')

# Connect to the server using the default namespace "/" and the proper socketio_path
sio.connect(server_url, namespaces=['/'], socketio_path='socket.io')

artnet_dummy = 0
al_features_dummy = 0
lx1_rgb_array = None
lx2_rgb_array = None
lx3_rgb_array = None
lx1_values_from_webUI = None
lx2_values_from_webUI = None
lx3_values_from_webUI = None

def init():
    ###
    global al_features_from_file
    global data_lighting_pkl
    global data_lighting_to_table_add_01
    global lx1_rgb_array
    global lx2_rgb_array
    global lx3_rgb_array
    global lx1_values_from_webUI
    global lx2_values_from_webUI
    global lx3_values_from_webUI
    global lx1_array_descision_dict
    global lx2_array_descision_dict
    global lx3_array_descision_dict
    global CONFIG
    # global lx1_values

    al_features_from_file = op('AL_loader_pkl')
    pkl_file_path_lighting = str(op('AL_pkl_name')[0,0])
    pkl_file_path_lighting_add_01 = str(op('AL_pkl_name')[3,0])
    
    with open(pkl_file_path_lighting, 'rb') as file:
        data_lighting_pkl = pickle.load(file)
        
    with open(pkl_file_path_lighting_add_01, 'rb') as file:
        data_lighting_to_table_add_01 = pickle.load(file)

    data_lighting_pkl = data_lighting_pkl[ :, :60]
    data_lighting_to_table_add_01 = data_lighting_to_table_add_01[ :, :60]
    # print(f'data_lighting_pkl.shape: {data_lighting_pkl.shape}')

    CONFIG = {
        "max_cycles_per_second": 4.0,        # Base number of cycles across LED spatial positions when freq=1.0
        "max_phase_cycles_per_second": 8.0,    # Maximum phase cycles per second
        "led_count": 33,                     # Physical LED count
        "virtual_led_count": 8,              # Virtual luminaire count for odd/even modulation (if needed)
        "fps": 30,                           # Frames per second
        "mode": "hard",                      # Mode for odd/even modulation: 'hard' or 'smooth'
        "bpm_thresholds": {
            "low": 80,    # BPM below this → slower movement
            "high": 135   # BPM above this → faster, more abrupt transitions
        },
        "optimization": {
            "alpha": 1.0,  # cost weight for matching maximum intensity
            "beta": 1.0,   # cost weight for matching minimum intensity
            "delta": 1.0   # cost weight for matching oscillation (number of cycles)
        },
        # A constant that can be used to influence the decision based on oscillatory behavior.
        "oscillation_threshold": 10,       # e.g. if more than 5 peaks are detected, consider it “high oscillation”
        "geo_phase_threshold": 0.15,       # expected phase range (max-min) for moderate variation
        "geo_freq_threshold": 0.15,        # expected frequency range for moderate variation
        "geo_offset_threshold": 0.15       # expected offset range for moderate variation
    }

    frames = data_lighting_pkl.shape[0]
    al_size_all = data_lighting_pkl.shape[1]

    al_GeoAppv01_len = 10
    # LED_COUNT = 33  # Number of RGB values to generate

    mirroring_active_lx1 = 0.0
    mirroring_active_lx2 = 0.0
    mirroring_active_lx3 = 0.0
    zero_amount_lx1 = 0.0
    zero_amount_lx2 = 0.0
    zero_amount_lx3 = 0.0
    moving_direction_lx1 = 0.0
    moving_direction_lx2 = 0.0
    moving_direction_lx3 = 0.0

    mirroring_active_lx1 = float(op('Mirror_On_Off_lx1')[0,0])
    moving_direction_lx1 = float(op('moving_direction_lx1')[0,0])
    highlights_lx1 = float(op('highlights_On_Off_lx1')[0,0])
    mirroring_active_lx2 = float(op('Mirror_On_Off_lx2')[0,0])
    moving_direction_lx2 = float(op('moving_direction_lx2')[0,0])
    highlights_lx2 = float(op('highlights_On_Off_lx2')[0,0])
    mirroring_active_lx3 = float(op('Mirror_On_Off_lx3')[0,0])
    moving_direction_lx3 = float(op('moving_direction_lx3')[0,0])
    highlights_lx3 = float(op('highlights_On_Off_lx3')[0,0])
    lead_group = float(op('Lead_Group')[0,0])

    lx1_standard_allframes = data_lighting_pkl[:, 0:10]
    lx1_highlight_allframes = data_lighting_pkl[:, 10:20]
    lx2_standard_allframes = data_lighting_pkl[:, 20:30]
    lx2_highlight_allframes = data_lighting_pkl[:, 30:40]
    lx3_standard_allframes = data_lighting_pkl[:, 40:50]
    lx3_highlight_allframes = data_lighting_pkl[:, 50:60]

    PASv02_to_lx1 = op('set_PASv02_LX1')[0,0]
    PASv02_to_lx2 = op('set_PASv02_LX2')[0,0]
    PASv02_to_lx3 = op('set_PASv02_LX3')[0,0]

    to_lx1_PASv02_allframes = data_lighting_to_table_add_01[:,PASv02_to_lx1 * 6 : (PASv02_to_lx1 + 1) * 6]
    to_lx2_PASv02_allframes = data_lighting_to_table_add_01[:,PASv02_to_lx2 * 6 : (PASv02_to_lx2 + 1) * 6]
    to_lx3_PASv02_allframes = data_lighting_to_table_add_01[:,PASv02_to_lx3 * 6 : (PASv02_to_lx3 + 1) * 6]

    # freq = al_GeoApproach_allframes[:, 4]

    # Features AL GeoApproachv01 #01 for each fixture
    # |:
    # 00: pan_activity, 01: tilt_activity, 02: wave_type_a
    # 03: wave_type_b, 04: freq, 05: amplitude, 06: offset, 
    # 07: phase, 08: col_hue, 09: col_sat
    # :|

    lx1_standard_allframes_amplitude = lx1_standard_allframes[:, 5]
    lx2_standard_allframes_amplitude = lx2_standard_allframes[:, 5]
    lx3_standard_allframes_amplitude = lx3_standard_allframes[:, 5]

    lx1_standard_allframes_wave_type = allframesofasegment_to_wavetype(lx1_standard_allframes)
    lx1_highlight_allframes_wave_type = allframesofasegment_to_wavetype(lx1_highlight_allframes)
    lx2_standard_allframes_wave_type = allframesofasegment_to_wavetype(lx2_standard_allframes)
    lx2_highlight_allframes_wave_type = allframesofasegment_to_wavetype(lx2_highlight_allframes)
    lx3_standard_allframes_wave_type = allframesofasegment_to_wavetype(lx3_standard_allframes)
    lx3_highlight_allframes_wave_type = allframesofasegment_to_wavetype(lx3_highlight_allframes)

    lx1_standard_allframes_freq = allframesofasegment_to_freq(lx1_standard_allframes)
    lx1_highlight_allframes_freq = allframesofasegment_to_freq(lx1_highlight_allframes)
    lx2_standard_allframes_freq = allframesofasegment_to_freq(lx2_standard_allframes)
    lx2_highlight_allframes_freq = allframesofasegment_to_freq(lx2_highlight_allframes)
    lx3_standard_allframes_freq = allframesofasegment_to_freq(lx3_standard_allframes)
    lx3_highlight_allframes_freq = allframesofasegment_to_freq(lx3_highlight_allframes)

    lx1_standard_allframes_pan_act, lx1_standard_allframes_tilt_act = allframesofasegment_to_pantilt_activity(lx1_standard_allframes)
    lx1_highlight_allframes_pan_act, lx1_highlight_allframes_tilt_act = allframesofasegment_to_pantilt_activity(lx1_highlight_allframes)
    lx2_standard_allframes_pan_act, lx2_standard_allframes_tilt_act = allframesofasegment_to_pantilt_activity(lx2_standard_allframes)
    lx2_highlight_allframes_pan_act, lx2_highlight_allframes_tilt_act = allframesofasegment_to_pantilt_activity(lx2_highlight_allframes)
    lx3_standard_allframes_pan_act, lx3_standard_allframes_tilt_act = allframesofasegment_to_pantilt_activity(lx3_standard_allframes)
    lx3_highlight_allframes_pan_act, lx3_highlight_allframes_tilt_act = allframesofasegment_to_pantilt_activity(lx3_highlight_allframes)

    lx1_standard_allframes_phase = allframesofasegment_to_phase(lx1_standard_allframes)
    lx1_highlight_allframes_phase = allframesofasegment_to_phase(lx1_highlight_allframes)
    lx2_standard_allframes_phase = allframesofasegment_to_phase(lx2_standard_allframes)
    lx2_highlight_allframes_phase = allframesofasegment_to_phase(lx2_highlight_allframes)
    lx3_standard_allframes_phase = allframesofasegment_to_phase(lx3_standard_allframes)
    lx3_highlight_allframes_phase = allframesofasegment_to_phase(lx3_highlight_allframes)

    lx1_standard_allframes_offset = allframesofasegment_to_offset(lx1_standard_allframes)
    lx1_highlight_allframes_offset = allframesofasegment_to_offset(lx1_highlight_allframes)
    lx2_standard_allframes_offset = allframesofasegment_to_offset(lx2_standard_allframes)
    lx2_highlight_allframes_offset = allframesofasegment_to_offset(lx2_highlight_allframes)
    lx3_standard_allframes_offset = allframesofasegment_to_offset(lx3_standard_allframes)
    lx3_highlight_allframes_offset = allframesofasegment_to_offset(lx3_highlight_allframes)

    lx1_standard_allframes_col_hue, lx1_standard_allframes_col_sat = allframesofasegment_to_colhue_colsat(lx1_standard_allframes)
    lx1_highlight_allframes_col_hue, lx1_highlight_allframes_col_sat = allframesofasegment_to_colhue_colsat(lx1_highlight_allframes)
    lx2_standard_allframes_col_hue, lx2_standard_allframes_col_sat = allframesofasegment_to_colhue_colsat(lx2_standard_allframes)
    lx2_highlight_allframes_col_hue, lx2_highlight_allframes_col_sat = allframesofasegment_to_colhue_colsat(lx2_highlight_allframes)
    lx3_standard_allframes_col_hue, lx3_standard_allframes_col_sat = allframesofasegment_to_colhue_colsat(lx3_standard_allframes)
    lx3_highlight_allframes_col_hue, lx3_highlight_allframes_col_sat = allframesofasegment_to_colhue_colsat(lx3_highlight_allframes)

    bpm_value = float(op('current_song_BPM')[0,0])

    lx1_standard_data_dict = {
    'standard_amplitude': lx1_standard_allframes_amplitude,
    'wave_type': lx1_standard_allframes_wave_type,
    'freq': lx1_standard_allframes_freq,
    'phase': lx1_standard_allframes_phase,
    'offset': lx1_standard_allframes_offset,
    'col_hue': lx1_standard_allframes_col_hue,
    'col_sat': lx1_standard_allframes_col_sat,
    'bpm': bpm_value,
    'frames': frames,
    'mirroring_active': mirroring_active_lx1,
    'zero_amount': zero_amount_lx1,
    'moving_direction': moving_direction_lx1,
    'current_lx_number' : 'lx1',
    'PASv02_to_lx' : PASv02_to_lx1,
    'PASv02_allframes': to_lx1_PASv02_allframes,
    'highlights': highlights_lx1
    }
    lx1_array_descision_dict = construct_array_decision(lx1_standard_data_dict, CONFIG)

    lx2_standard_data_dict = {
    'standard_amplitude': lx2_standard_allframes_amplitude,
    'wave_type': lx2_standard_allframes_wave_type,
    'freq': lx2_standard_allframes_freq,
    'phase': lx2_standard_allframes_phase,
    'offset': lx2_standard_allframes_offset,
    'col_hue': lx2_standard_allframes_col_hue,
    'col_sat': lx2_standard_allframes_col_sat,
    'bpm': bpm_value,
    'frames': frames,
    'mirroring_active': mirroring_active_lx2,
    'zero_amount': zero_amount_lx2,
    'moving_direction': moving_direction_lx2,
    'current_lx_number' : 'lx2',
    'PASv02_to_lx' : PASv02_to_lx2,
    'PASv02_allframes': to_lx2_PASv02_allframes,
    'highlights': highlights_lx2
    }
    lx2_array_descision_dict = construct_array_decision(lx2_standard_data_dict, CONFIG)

    lx3_standard_data_dict = {
    'standard_amplitude': lx3_standard_allframes_amplitude,
    'wave_type': lx3_standard_allframes_wave_type,
    'freq': lx3_standard_allframes_freq,
    'phase': lx3_standard_allframes_phase,
    'offset': lx3_standard_allframes_offset,
    'col_hue': lx3_standard_allframes_col_hue,
    'col_sat': lx3_standard_allframes_col_sat,
    'bpm': bpm_value,
    'frames': frames,
    'mirroring_active': mirroring_active_lx3,
    'zero_amount': zero_amount_lx3,
    'moving_direction': moving_direction_lx3,
    'current_lx_number' : 'lx3',
    'PASv02_to_lx' : PASv02_to_lx3,
    'PASv02_allframes': to_lx3_PASv02_allframes,
    'highlights': highlights_lx3
    }
    lx3_array_descision_dict = construct_array_decision(lx3_standard_data_dict, CONFIG)

    lx1_wavetype_decision = lx1_array_descision_dict['decision']
    lx1_overall_dynamic = lx1_array_descision_dict['overall_dynamic']
    lx1_f0 = lx1_array_descision_dict['f0']
    lx1_phase_movement = lx1_array_descision_dict['phase_movement']
    lx1_col_hue = lx1_array_descision_dict['col_hue']
    lx1_col_sat = lx1_array_descision_dict['col_sat']
    lx1_mirroring_active = lx1_array_descision_dict['mirroring_active']
    lx1_moving_direction = lx1_array_descision_dict['moving_direction']
    lx1_highlights = lx1_array_descision_dict['highlights']

    lx2_wavetype_decision = lx2_array_descision_dict['decision']
    lx2_overall_dynamic = lx2_array_descision_dict['overall_dynamic']
    lx2_f0 = lx2_array_descision_dict['f0']
    lx2_phase_movement = lx2_array_descision_dict['phase_movement']
    lx2_col_hue = lx2_array_descision_dict['col_hue']
    lx2_col_sat = lx2_array_descision_dict['col_sat']
    lx2_mirroring_active = lx2_array_descision_dict['mirroring_active']
    lx2_moving_direction = lx2_array_descision_dict['moving_direction']
    lx2_highlights = lx2_array_descision_dict['highlights']

    lx3_wavetype_decision = lx3_array_descision_dict['decision']
    lx3_overall_dynamic = lx3_array_descision_dict['overall_dynamic']
    lx3_f0 = lx3_array_descision_dict['f0']
    lx3_phase_movement = lx3_array_descision_dict['phase_movement']
    lx3_col_hue = lx3_array_descision_dict['col_hue']
    lx3_col_sat = lx3_array_descision_dict['col_sat']
    lx3_mirroring_active = lx3_array_descision_dict['mirroring_active']
    lx3_moving_direction = lx3_array_descision_dict['moving_direction']
    lx3_highlights = lx3_array_descision_dict['highlights']

    ###
    if lead_group == 0.0:
        print('Led Lead Grouping passed')
        lx1_create_variance = 0.0
        lx2_create_variance = 0.0
        lx3_create_variance = 0.0
        lead_group_str = 'Lead_Grouping_bypass'
        pass
    else:
        print('Led Lead Grouping active')
        if lead_group == 1.0:
            lx2_wavetype_decision = lx1_wavetype_decision
            lx2_overall_dynamic = lx1_overall_dynamic
            lx2_f0 = lx1_f0
            lx2_phase_movement = lx1_phase_movement
            lx2_col_hue = lx1_col_hue
            lx2_col_sat = lx1_col_sat

            lx3_wavetype_decision = lx1_wavetype_decision
            lx3_overall_dynamic = lx1_overall_dynamic
            lx3_f0 = lx1_f0
            lx3_phase_movement = lx1_phase_movement
            lx3_col_hue = lx1_col_hue
            lx3_col_sat = lx1_col_sat

            lx1_create_variance = 0.0
            lx2_create_variance = 1.0
            lx3_create_variance = 1.0

            lead_group_str = 'Lead_Grouping_Master_LX1'

        elif lead_group == 2.0:
            lx1_wavetype_decision = lx2_wavetype_decision
            lx1_overall_dynamic = lx2_overall_dynamic
            lx1_f0 = lx2_f0
            lx1_phase_movement = lx2_phase_movement
            lx1_col_hue = lx2_col_hue
            lx1_col_sat = lx2_col_sat

            lx3_wavetype_decision = lx2_wavetype_decision
            lx3_overall_dynamic = lx2_overall_dynamic
            lx3_f0 = lx2_f0
            lx3_phase_movement = lx2_phase_movement
            lx3_col_hue = lx2_col_hue
            lx3_col_sat = lx2_col_sat

            lx1_create_variance = 1.0
            lx2_create_variance = 0.0
            lx3_create_variance = 1.0

            lead_group_str = 'Lead_Grouping_Master_LX2'

        elif lead_group == 3.0: 
            lx1_wavetype_decision = lx3_wavetype_decision
            lx1_overall_dynamic = lx3_overall_dynamic
            lx1_f0 = lx3_f0
            lx1_phase_movement = lx3_phase_movement
            lx1_col_hue = lx3_col_hue
            lx1_col_sat = lx3_col_sat

            lx2_wavetype_decision = lx3_wavetype_decision
            lx2_overall_dynamic = lx3_overall_dynamic
            lx2_f0 = lx3_f0
            lx2_phase_movement = lx3_phase_movement
            lx2_col_hue = lx3_col_hue
            lx2_col_sat = lx3_col_sat

            lx1_create_variance = 1.0
            lx2_create_variance = 1.0
            lx3_create_variance = 0.0

            lead_group_str = 'Lead_Grouping_Master_LX3'


    # Generating Defaults
    lx1_movement = 0.5
    lx2_movement = 0.5
    lx3_movement = 0.5
    lx1_position = "Pos01"
    lx2_position = "Pos01"
    lx3_position = "Pos01"

    # Add Logic to read and write Movement and Positions

    lx1_array_descision_dict = {
        'decision': lx1_wavetype_decision,
        'overall_dynamic': lx1_overall_dynamic,
        'f0': lx1_f0,
        'phase_movement': lx1_phase_movement,
        'col_hue' : lx1_col_hue,
        'col_sat' : lx1_col_sat,
        'create_variance' : lx1_create_variance,
        'lead_group_str' : lead_group_str,
        'lx_movement' : lx1_movement,
        'lx_position' : lx1_position,
        'mirroring_active' : lx1_mirroring_active,
        'moving_direction' : lx1_moving_direction,
        'highlights' : lx1_highlights
    }

    lx2_array_descision_dict = {    
        'decision': lx2_wavetype_decision,
        'overall_dynamic': lx2_overall_dynamic,
        'f0': lx2_f0,
        'phase_movement': lx2_phase_movement,
        'col_hue' : lx2_col_hue,
        'col_sat' : lx2_col_sat,
        'create_variance' : lx2_create_variance,
        'lead_group_str' : lead_group_str,
        'lx_movement' : lx2_movement,
        'lx_position' : lx2_position,
        'mirroring_active' : lx2_mirroring_active,
        'moving_direction' : lx2_moving_direction,
        'highlights' : lx2_highlights
    }

    lx3_array_descision_dict = {    
        'decision': lx3_wavetype_decision,
        'overall_dynamic': lx3_overall_dynamic,
        'f0': lx3_f0,
        'phase_movement': lx3_phase_movement,
        'col_hue' : lx3_col_hue,
        'col_sat' : lx3_col_sat,
        'create_variance' : lx3_create_variance,
        'lead_group_str' : lead_group_str,
        'lx_movement' : lx3_movement,
        'lx_position' : lx3_position,
        'mirroring_active' : lx3_mirroring_active,
        'moving_direction' : lx3_moving_direction,
        'highlights' : lx3_highlights
    }

    ###

    try:
        lx1_values_from_webUI = shared.global_lx1_webUI_values
        lx2_values_from_webUI = shared.global_lx2_webUI_values
        lx3_values_from_webUI = shared.global_lx3_webUI_values
        webUI_data_received = True
    except AttributeError:
        print("WebUI values not yet received from server. Using initial default values.")
        lx1_values_from_webUI = None
        lx2_values_from_webUI = None
        lx3_values_from_webUI = None
        webUI_data_received = False

    # phase_movement_from_webUI = 1 if moving_direction < 0.5 else -1

    # decision_webUI_lx1 = lx1_values_from_webUI['decision']
    # print(f'decision_webUI_lx1: {decision_webUI_lx1}')

    webUI_in_charge_toggle = op('webui_in_charge_toggle')[0,0]

    if webUI_in_charge_toggle == 1.0:
        print('WebUI in charge')
        lx1_values_from_webUI_moving_direction = lx1_values_from_webUI['moving_direction']
        lx2_values_from_webUI_moving_direction = lx2_values_from_webUI['moving_direction']
        lx3_values_from_webUI_moving_direction = lx3_values_from_webUI['moving_direction']

        lx1_values_from_webUI_phase_movement = lx1_values_from_webUI['phase_movement']
        lx2_values_from_webUI_phase_movement = lx2_values_from_webUI['phase_movement']
        lx3_values_from_webUI_phase_movement = lx3_values_from_webUI['phase_movement']

        if lx1_values_from_webUI_moving_direction < 0.5:
            lx1_values_from_webUI_phase_movement = lx1_values_from_webUI_phase_movement * -1
        if lx2_values_from_webUI_moving_direction < 0.5:
            lx2_values_from_webUI_phase_movement = lx2_values_from_webUI_phase_movement * -1
        if lx3_values_from_webUI_moving_direction < 0.5:
            lx3_values_from_webUI_phase_movement = lx3_values_from_webUI_phase_movement * -1

        lx1_values_from_webUI['phase_movement'] = lx1_values_from_webUI_phase_movement
        lx2_values_from_webUI['phase_movement'] = lx2_values_from_webUI_phase_movement
        lx3_values_from_webUI['phase_movement'] = lx3_values_from_webUI_phase_movement

        lx1_array_descision_dict = lx1_values_from_webUI
        lx2_array_descision_dict = lx2_values_from_webUI
        lx3_array_descision_dict = lx3_values_from_webUI

    elif webUI_in_charge_toggle == 0.0:
        print('WebUI not in charge')

    lx1_highlights = lx1_array_descision_dict['highlights']
    lx2_highlights = lx2_array_descision_dict['highlights']
    lx3_highlights = lx3_array_descision_dict['highlights']

    lx1_highlights_timings = get_highlight_time_positions_out_of_segment(lx1_highlight_allframes, frames, CONFIG)
    lx2_highlights_timings = get_highlight_time_positions_out_of_segment(lx2_highlight_allframes, frames, CONFIG)
    lx3_highlights_timings = get_highlight_time_positions_out_of_segment(lx3_highlight_allframes, frames, CONFIG)

    lx1_rgb_array = construct_rgb_array(lx1_standard_data_dict, lx1_array_descision_dict, CONFIG, lx1_highlights_timings)
    lx2_rgb_array = construct_rgb_array(lx2_standard_data_dict, lx2_array_descision_dict, CONFIG, lx2_highlights_timings)
    lx3_rgb_array = construct_rgb_array(lx3_standard_data_dict, lx3_array_descision_dict, CONFIG, lx3_highlights_timings)

    lx1_highlight_data_dict = {
    'rgb_standard_array': lx1_rgb_array,
    'highlight_allframes': lx1_highlight_allframes,
    'wave_type': lx1_highlight_allframes_wave_type,
    'freq': lx1_highlight_allframes_freq,
    'phase': lx1_highlight_allframes_phase,
    'offset': lx1_highlight_allframes_offset,
    'col_hue': lx1_highlight_allframes_col_hue,
    'col_sat': lx1_highlight_allframes_col_sat,
    'bpm': bpm_value,
    'frames': frames,
    'highlights': highlights_lx1
    }
    lx1_rgb_array = modify_rgb_array_with_highlight_segment(lx1_highlight_data_dict, CONFIG)
    lx1_rgb_array = np.tile(lx1_rgb_array, (20, 1, 1))[:2700]

    lx2_highlight_data_dict = {
    'rgb_standard_array': lx2_rgb_array,
    'highlight_allframes': lx2_highlight_allframes,
    'wave_type': lx2_highlight_allframes_wave_type,
    'freq': lx2_highlight_allframes_freq,
    'phase': lx2_highlight_allframes_phase,
    'offset': lx2_highlight_allframes_offset,
    'col_hue': lx2_highlight_allframes_col_hue,
    'col_sat': lx2_highlight_allframes_col_sat,
    'bpm': bpm_value,
    'frames': frames,
    'highlights': highlights_lx2
    }
    lx2_rgb_array = modify_rgb_array_with_highlight_segment(lx2_highlight_data_dict, CONFIG)
    lx2_rgb_array = np.tile(lx2_rgb_array, (20, 1, 1))[:2700]

    lx3_highlight_data_dict = {
    'rgb_standard_array': lx3_rgb_array,
    'highlight_allframes': lx3_highlight_allframes,
    'wave_type': lx3_highlight_allframes_wave_type,
    'freq': lx3_highlight_allframes_freq,
    'phase': lx3_highlight_allframes_phase,
    'offset': lx3_highlight_allframes_offset,
    'col_hue': lx3_highlight_allframes_col_hue,
    'col_sat': lx3_highlight_allframes_col_sat,
    'bpm': bpm_value,
    'frames': frames,
    'highlights': highlights_lx3
    }
    lx3_rgb_array = modify_rgb_array_with_highlight_segment(lx3_highlight_data_dict, CONFIG)
    lx3_rgb_array = np.tile(lx3_rgb_array, (20, 1, 1))[:2700]

    # print(f'lx1_rgb_array.shape: {lx1_rgb_array.shape}')
    # print(f'current song: {pkl_file_path_lighting}')
    print(f'current song BPM: {bpm_value}')
    print('init done')

    return al_features_from_file, data_lighting_pkl


#######################

def whileOn(channel, sampleIndex, val, prev):
    global lx1_rgb_array
    global lx2_rgb_array
    global lx3_rgb_array
    global artnet_dummy
    global al_features_dummy
    global al_features_from_file
    global data_lighting_pkl

    # config_table = op('config_table')
    # len_16uni = 8192
    len_unis = 512
    init_pulse = op('init')[0, 0]
    if init_pulse == 1:
        init()

    len_al_all = 60
    current_frame = int(me.time.frame) - 1
    al_features_dummy = np.zeros((1, len_al_all))

    # Change this later, so that it will work without a loop
    for i in range(len_al_all):
        al_features_dummy[0, i] = al_features_from_file[current_frame, i]

    AL_to_artnet_uni1, AL_to_artnet_uni2  = features_60paramsgeoapproach_to_artnet_v01(al_features_dummy)
    # print(AL_to_artnet[:,512*11])
    for i in range(len_unis):
        op('artnet_after_backconversion_small_uni1')[0, i] = AL_to_artnet_uni1[0, i] * 255
        op('artnet_after_backconversion_small_uni2')[0, i] = AL_to_artnet_uni2[0, i] * 255
    return

#######################

def features_60paramsgeoapproach_to_artnet_v01(features_v04_currentframe):
    global lx1_rgb_array
    global lx2_rgb_array
    global lx3_rgb_array
    global lx1_array_descision_dict
    global lx2_array_descision_dict
    global lx3_array_descision_dict
    global artnet_dummy
    global al_features_dummy
    global al_features_from_file
    global data_lighting_pkl
    global CONFIG

    len = 1
    one_universe_size = 512
    AL_to_artnet_uni1 = np.zeros((len, one_universe_size))
    AL_to_artnet_uni2 = np.zeros((len, one_universe_size))
    led_wash_group_truss_count = 7
    one_wash_target_size = 7
    LED_COUNT = 33  # Number of RGB values to generate

    current_frame = int(me.time.frame) - 1

    lx1_rgb_array_current_frame = lx1_rgb_array[current_frame]
    lx2_rgb_array_current_frame = lx2_rgb_array[current_frame]
    lx3_rgb_array_current_frame = lx3_rgb_array[current_frame]

    virtual_led_count = CONFIG["virtual_led_count"]
    LED_COUNT = CONFIG["led_count"]
    if LED_COUNT % 2 == 0:
        indices = np.linspace(1, (LED_COUNT - 2), virtual_led_count, dtype=int)
    else:
        indices = np.linspace(1, (LED_COUNT - 3), virtual_led_count, dtype=int)

    # print(f'indices: {indices}')

    lx1_rgb_redux_array_current_frame = lx1_rgb_array_current_frame[indices]
    lx2_rgb_redux_array_current_frame = lx2_rgb_array_current_frame[indices]
    lx3_rgb_redux_array_current_frame = lx3_rgb_array_current_frame[indices]

    for i in range(LED_COUNT):
        op('rgb_array_standard_lx1')[0,i] = lx1_rgb_array_current_frame[i,0]  # R value
        op('rgb_array_standard_lx1')[1,i] = lx1_rgb_array_current_frame[i,1]  # G value
        op('rgb_array_standard_lx1')[2,i] = lx1_rgb_array_current_frame[i,2]  # B value

        op('rgb_array_standard_lx2')[0,i] = lx2_rgb_array_current_frame[i,0]  # R value
        op('rgb_array_standard_lx2')[1,i] = lx2_rgb_array_current_frame[i,1]  # G value
        op('rgb_array_standard_lx2')[2,i] = lx2_rgb_array_current_frame[i,2]  # B value

        op('rgb_array_standard_lx3')[0,i] = lx3_rgb_array_current_frame[i,0]  # R value
        op('rgb_array_standard_lx3')[1,i] = lx3_rgb_array_current_frame[i,1]  # G value
        op('rgb_array_standard_lx3')[2,i] = lx3_rgb_array_current_frame[i,2]  # B value

    for i in range(virtual_led_count):
        op('rgb_array_standard_redux_lx1')[0,i] = lx1_rgb_redux_array_current_frame[i,0]  # R value
        op('rgb_array_standard_redux_lx1')[1,i] = lx1_rgb_redux_array_current_frame[i,1]  # G value
        op('rgb_array_standard_redux_lx1')[2,i] = lx1_rgb_redux_array_current_frame[i,2]  # B value

    pan_val = 0.5
    # Cuete_Tilt_LX1 = op('Cuete_Tilt_LX1')[0,0]
    # Cuete_Tilt_LX2 = op('Cuete_Tilt_LX2')[0,0]
    LEDBeam_Tilt_LX3 = op('LEDBeam_Tilt_LX3')[0,0]

    Position_LX1 = lx1_array_descision_dict['lx_position']
    Position_LX2 = lx2_array_descision_dict['lx_position']
    Position_LX3 = lx3_array_descision_dict['lx_position']

    if Position_LX1 == 'Pos01':
        op('POS_LX1_Switch_Index')[0,0] = 0
        op('LX_Movement_Multiplicator')[0,0] = 2
    elif Position_LX1 == 'Pos02':
        op('POS_LX1_Switch_Index')[0,0] = 1
        op('LX_Movement_Multiplicator')[0,0] = 2
    elif Position_LX1 == 'Pos03':
        op('POS_LX1_Switch_Index')[0,0] = 1
        op('LX_Movement_Multiplicator')[0,0] = 1
    elif Position_LX1 == 'Pos04':
        op('POS_LX1_Switch_Index')[0,0] = 2
        op('LX_Movement_Multiplicator')[0,0] = 1
    else:
        op('POS_LX1_Switch_Index')[0,0] = 0
        op('LX_Movement_Multiplicator')[0,0] = 2

    if Position_LX2 == 'Pos01':
        op('POS_LX2_Switch_Index')[0,0] = 0
        op('LX_Movement_Multiplicator')[0,0] = 2
    elif Position_LX2 == 'Pos02':
        op('POS_LX2_Switch_Index')[0,0] = 1
        op('LX_Movement_Multiplicator')[0,0] = 2
    elif Position_LX1 == 'Pos03':
        op('POS_LX2_Switch_Index')[0,0] = 1
        op('LX_Movement_Multiplicator')[0,0] = 1
    elif Position_LX1 == 'Pos04':
        op('POS_LX2_Switch_Index')[0,0] = 2
        op('LX_Movement_Multiplicator')[0,0] = 1
    else:
        op('POS_LX2_Switch_Index')[0,0] = 0
        op('LX_Movement_Multiplicator')[0,0] = 2

    if Position_LX3 == 'Pos01':
        op('POS_LX3_Switch_Index')[0,0] = 0
        op('LX_Movement_Multiplicator')[0,0] = 2
    elif Position_LX3 == 'Pos02':
        op('POS_LX3_Switch_Index')[0,0] = 1
        op('LX_Movement_Multiplicator')[0,0] = 2
    elif Position_LX3 == 'Pos03':
        op('POS_LX3_Switch_Index')[0,0] = 1
        op('LX_Movement_Multiplicator')[0,0] = 1
    elif Position_LX3 == 'Pos04':
        op('POS_LX3_Switch_Index')[0,0] = 2
        op('LX_Movement_Multiplicator')[0,0] = 1
    else:
        op('POS_LX3_Switch_Index')[0,0] = 0
        op('LX_Movement_Multiplicator')[0,0] = 2

    Cuete_Tilt_Array_LX1 = np.zeros((virtual_led_count, 1))
    Cuete_Pan_Array_LX1 = np.zeros((virtual_led_count, 1))
    Cuete_Tilt_Array_LX2 = np.zeros((virtual_led_count, 1))
    Cuete_Pan_Array_LX2 = np.zeros((virtual_led_count, 1))
    LEDBeam_Tilt_Array_LX3 = np.zeros((virtual_led_count, 1))
    LEDBeam_Pan_Array_LX3 = np.zeros((virtual_led_count, 1))

    for i in range(virtual_led_count):
        Cuete_Tilt_Array_LX1[i, 0] = op('POS_Table_LX1')[(i * 2 + 1), 1]
        # print('Cuete_Tilt_Array_LX1[i, 0]: ', Cuete_Tilt_Array_LX1[i, 0])
        Cuete_Pan_Array_LX1[i, 0] = op('POS_Table_LX1')[(i * 2), 1]
        # print('Cuete_Pan_Array_LX1[i, 0]: ', Cuete_Pan_Array_LX1[i, 0])
        Cuete_Tilt_Array_LX2[i, 0] = op('POS_Table_LX2')[(i * 2 + 1), 1]
        Cuete_Pan_Array_LX2[i, 0] = op('POS_Table_LX2')[(i * 2), 1]
        LEDBeam_Tilt_Array_LX3[i, 0] = op('POS_Table_LX3')[(i * 2 + 1), 1]
        LEDBeam_Pan_Array_LX3[i, 0] = op('POS_Table_LX3')[(i * 2), 1]
    
    lx1_AlphaBeam = RGB_array_to_AlphaBeam(lx1_rgb_redux_array_current_frame, Cuete_Pan_Array_LX1, Cuete_Tilt_Array_LX1, virtual_led_count)
    lx2_AlphaBeam = RGB_array_to_AlphaBeam(lx2_rgb_redux_array_current_frame, Cuete_Pan_Array_LX2, Cuete_Tilt_Array_LX2, virtual_led_count)
    lx3_AlphaBeam = RGB_array_to_AlphaBeam(lx3_rgb_redux_array_current_frame, LEDBeam_Pan_Array_LX3, LEDBeam_Tilt_Array_LX3, virtual_led_count)

    lx3_ColorFusion = RGB_array_to_ColorFusion(lx3_rgb_redux_array_current_frame)

    to_visualizer_uni1 = np.concatenate((lx1_AlphaBeam, lx2_AlphaBeam, lx3_ColorFusion), axis=0)
    to_visualizer_uni2 = lx3_AlphaBeam
    # print(f'to_visualizer.shape: {to_visualizer.shape}')
    # print(f'AL_to_artnet.shape: {AL_to_artnet.shape}')
    AL_to_artnet_uni1[0, :to_visualizer_uni1.shape[0]] = to_visualizer_uni1
    AL_to_artnet_uni2[0, :to_visualizer_uni2.shape[0]] = to_visualizer_uni2
    
    # AL_to_artnet = led_wash_fixture_specific_to_artnet(AL_to_artnet, led_wash_fixture_specific_lx1_truss, led_wash_group_truss_count, led_wash_group_truss_count * one_wash_target_size * 0)

    return AL_to_artnet_uni1, AL_to_artnet_uni2

#######################

def select_waveform_for_segment(luminaire_dict, config):
    """
    Extended decision function that uses both PAS and Geo approach parameters.
    
    PAS-based metrics (from luminaire_dict['PASv02_allframes']):
      - target_max: maximum intensity from PAS (column 0)
      - target_min: minimum inverse minima from PAS (column 3)
      - oscillation_count: number of peaks in the PAS intensity signal (a measure of how rapidly values change)
    
    Geo-based metrics (from luminaire_dict):
      - phase_range: variation of the phase over time (max - min)
      - freq_range: variation of the frequency over time (max - min)
      - offset_range: variation of the offset over time (max - min), if available
    
    Each geo metric is normalized by a configured threshold, and the average of these yields a geo dynamic score.
    Separately, the PAS oscillation count is normalized by its threshold.
    
    Then, an overall dynamic score is computed as the average of the geo dynamic score and the PAS dynamic score.

    """
    # ---------------------------
    # PAS-based Metrics:
    # ---------------------------
    PAS_all = luminaire_dict['PASv02_allframes']
    intensityPeakPAS = PAS_all[:, 0]
    intensityInverseMinimaPAS = PAS_all[:, 3]
    
    # Count peaks in PAS intensity as a measure of oscillation.
    peaks, _ = find_peaks(intensityPeakPAS, height=0.6)  # the height threshold is empirical ### !!! ### CHECK THIS
    oscillation_count = len(peaks)
    
    target_max = np.max(intensityPeakPAS)
    target_min = 1.0 - np.max(intensityInverseMinimaPAS)
    if target_min > target_max:
        intensity_range = target_max
    else:
        intensity_range = target_max - target_min

    # Normalize PAS oscillation (a higher count indicates more oscillation)
    pas_dynamic_score = oscillation_count / config["oscillation_threshold"]

    # Modify target_max and target_min to be in a not to small range
    amplitude_geo = luminaire_dict['standard_amplitude']
    target_max_geo = np.max(amplitude_geo)
    target_min_geo = np.min(amplitude_geo)
    if target_max_geo > target_max:
        target_max_modified = target_max_geo
    else:
        target_max_modified = target_max

    if target_min_geo < target_min:
        target_min_modified = target_min_geo
    else:    
        target_min_modified = target_min

    # ---------------------------
    # Geo-based Metrics:
    # ---------------------------
    # Phase variation
    phase_geo = luminaire_dict['phase']
    geo_phase_range = np.max(phase_geo) - np.min(phase_geo)
    geo_phase_norm = geo_phase_range / config.get("geo_phase_threshold", 0.15)
    
    # Frequency variation: if provided as an array.
    freq_geo = luminaire_dict.get('freq')
    if freq_geo is not None and hasattr(freq_geo, '__len__') and np.ndim(freq_geo) > 0:
        geo_freq_range = np.max(freq_geo) - np.min(freq_geo)
    else:
        # If not available or scalar, assume minimal variation.
        geo_freq_range = 0.0
    geo_freq_norm = geo_freq_range / config.get("geo_freq_threshold", 0.15)
    
    # Offset variation: if provided (optional)
    offset_geo = luminaire_dict.get('offset')
    if offset_geo is not None and hasattr(offset_geo, '__len__') and np.ndim(offset_geo) > 0:
        geo_offset_range = np.max(offset_geo) - np.min(offset_geo)
    else:
        geo_offset_range = 0.0
    geo_offset_norm = geo_offset_range / config.get("geo_offset_threshold", 0.15)
    
    # Overall geo dynamic score as the average of normalized geo metrics.
    overall_geo_dynamic = (geo_phase_norm + geo_freq_norm + geo_offset_norm) / 3.0

    # ---------------------------
    # BPM:
    # ---------------------------
    bpm = luminaire_dict.get('bpm', 120)

    # ---------------------------
    # Combine PAS and Geo Dynamics:
    # ---------------------------
    overall_dynamic = (overall_geo_dynamic + pas_dynamic_score) / 2.0

    # ---------------------------
    # Decision Rules:
    # ---------------------------
    decision_boundary_01 = float(op('decision_boundary_01')[0,0])
    decision_boundary_02 = float(op('decision_boundary_02')[0,0]) 
    decision_boundary_03 = float(op('decision_boundary_03')[0,0])
    decision_boundary_04 = float(op('decision_boundary_04')[0,0])
    decision_boundary_05 = float(op('decision_boundary_05')[0,0])
    decision = None
    if intensity_range < decision_boundary_01:
        decision = "still"
    else:
        if overall_dynamic < decision_boundary_02:
            decision = "sine"
        elif overall_dynamic < decision_boundary_03:
            decision = "pwm_basic"
        elif overall_dynamic < decision_boundary_04:
            decision = "pwm_extended"
        elif overall_dynamic < decision_boundary_05:
            decision = "odd_even"
        else:
            # For very high overall dynamics, choose "square" if BPM is high,
            # otherwise, choose "odd_even" to let the odd/even modulation be driven by the beat.
            if bpm > config['bpm_thresholds']['high']:
                decision = "square"
            else:
                decision = "random"
    
    # ---------------------------
    # Print out computed metrics:
    # ---------------------------
    print("Waveform decision for this segment:", decision)
    # print(f"  BPM: {bpm}")
    print(f"  PAS Intensity Range: {intensity_range:.2f}  (max: {target_max:.2f}, min: {target_min:.2f})")
    # print(f"  Geo Phase Range: {geo_phase_range:.2f}  (Normalized: {geo_phase_norm:.2f})")
    # print(f"  Geo Frequency Range: {geo_freq_range:.2f}  (Normalized: {geo_freq_norm:.2f})")
    # print(f"  Geo Offset Range: {geo_offset_range:.2f}  (Normalized: {geo_offset_norm:.2f})")
    # print('   --')
    # print(f"  Normalized PAS Score: {pas_dynamic_score:.2f}  PAS Oscillation Count: {oscillation_count}  ")
    # print(f"  Overall Geo Dynamic Score: {overall_geo_dynamic:.2f}")
    print(f"  Overall Dynamic Score (Geo + PAS): {overall_dynamic:.2f}")
    
    return decision, overall_dynamic 

#######################

def construct_array_decision(luminaire_dict, config):
    """
    For a given musical segment (in luminaire_dict) decide on one waveform generator type,
    optimize its parameters to match target max/min and oscillation characteristics, print the decision,
    and then generate an RGB array for the segment.
    """

    frames = luminaire_dict['frames']
    bpm = luminaire_dict.get('bpm', 120)  # default if not provided
    
    # Geo parameters (for color and frequency)
    freq_geo = luminaire_dict['freq']          # base frequency from geo (a scalar)
    phase_geo = luminaire_dict['phase']        # array over time; we use its mean for phase movement
    col_hue_Geo = luminaire_dict['col_hue']
    col_sat_Geo = luminaire_dict['col_sat']
    col_hue_PAS = ((np.mean(luminaire_dict['PASv02_allframes'][:, 4])) / 1.0)
    col_sat_PAS = ((np.mean(luminaire_dict['PASv02_allframes'][:, 5])) / 1.0)
    mirroring_active = luminaire_dict['mirroring_active']
    moving_direction = luminaire_dict['moving_direction']
    highlights = luminaire_dict['highlights']

    col_dict = {
        'col_hue_GEO' : col_hue_Geo,
        'col_sat_GEO' : col_sat_Geo,
        'col_hue_PAS' : col_hue_PAS,
        'col_sat_PAS' : col_sat_PAS
    }

    col_hue, col_sat = set_color_for_segment(col_dict)

    if bpm < config['bpm_thresholds']['low']:
        bpm_scale = 0.5  # slower movement
    elif bpm > config['bpm_thresholds']['high']:
        bpm_scale = 2.0  # faster movement
    else:
        bpm_scale = 1.0
    f0 = freq_geo * bpm_scale  # base frequency adjusted by BPM

    mean_phase = np.mean(phase_geo)
    phase_direction = 1 if moving_direction < 0.5 else -1
    phase_movement = (mean_phase * config["max_phase_cycles_per_second"] / frames) * phase_direction


    decision, overall_dynamic = select_waveform_for_segment(luminaire_dict, config)

    array_descision_dict = {
        'decision': decision,
        'overall_dynamic': overall_dynamic,
        'f0': f0,
        'phase_movement': phase_movement,
        'col_hue' : col_hue,
        'col_sat' : col_sat,
        'mirroring_active' : mirroring_active,
        'moving_direction' : moving_direction,
        'highlights' : highlights
    }

    return array_descision_dict

#######################

def construct_rgb_array(luminaire_dict, array_descision_dict, config, lx_highlights_timings):

    LED_COUNT = config["led_count"]
    fps = config["fps"]
    virtual_led_count = config["virtual_led_count"]
    max_cycles = config["max_cycles_per_second"]
    effective_led_count = LED_COUNT

    mirroring_active = array_descision_dict["mirroring_active"]
    moving_direction = array_descision_dict["moving_direction"]
    frames = luminaire_dict['frames']
    bpm = luminaire_dict.get('bpm', 120)  # default if not provided
    current_lx_number = luminaire_dict['current_lx_number']

    decision = array_descision_dict['decision']
    overall_dynamic = array_descision_dict['overall_dynamic']
    f0 = array_descision_dict['f0']
    phase_movement = array_descision_dict['phase_movement']
    col_hue = array_descision_dict['col_hue']
    col_sat = array_descision_dict['col_sat']
    create_variance = array_descision_dict['create_variance']
    song_part_name = str((op('AL_pkl_name')[2,0]))
    current_lx_number = str(luminaire_dict['current_lx_number'])

    base_dir = '/Users/chetbaker/work/i_0002_MaschinenLicht/36_software/TD/GeoApproach/Touchdesigner/'
    # print(base_dir)
    relative_path_to_store_the_JSONs = os.path.join(base_dir, "assets_GeoApproach", "AudioSegment_LightControl_JSONs")

    data_exchange_to_webserver = write_variables_to_json(config, luminaire_dict, array_descision_dict, song_part_name, lx_highlights_timings, relative_path_to_store_the_JSONs)
    
    # --- Send Feature Updates ---
    send_feature_update(data_exchange_to_webserver)

    if current_lx_number == "lx1":
        data_exchange_to_webserver_lx1 = data_exchange_to_webserver
    elif current_lx_number == "lx2":
        data_exchange_to_webserver_lx2 = data_exchange_to_webserver
    elif current_lx_number == "lx3":
        data_exchange_to_webserver_lx3 = data_exchange_to_webserver
    else:
        pass


    def sine_basis(x, phase_offset, freq_adj, virtual_led_count, create_variance):
        # Use (virtual_led_count - 2) to determine the number of peaks
        peaks = (virtual_led_count - 4) / 2  # divide by 2 because sine wave has peaks and troughs
        if create_variance == 0.0:
            pass
        else:
            phase_offset = phase_offset + (create_variance * 0.33)
        return 0.5 + 0.5 * np.sin(2 * np.pi * (freq_adj * peaks * x + phase_offset))

    
    def square_basis(x, phase_offset, freq_adj, virtual_led_count, create_variance, divider=32):
        # Use virtual_led_count to determine the number of squares
        # This will create squares that align with virtual LED positions
        squares_per_strip = virtual_led_count / divider
        if create_variance == 0.0:
            pass
        else:
            phase_offset = phase_offset + create_variance
        return (np.mod(freq_adj * squares_per_strip * x + phase_offset, 1) < 0.5).astype(float)

    
    def saw_up_basis(x, phase_offset, freq_adj, virtual_led_count, create_variance):
        if create_variance == 0.0:
            pass
        else:
            phase_offset = phase_offset + create_variance
        return np.mod(freq_adj * max_cycles * x + phase_offset, 1)
    
    def saw_down_basis(x, phase_offset, freq_adj, virtual_led_count, create_variance):
        if create_variance == 0.0:
            pass
        else:
            phase_offset = phase_offset + create_variance
        return 1 - np.mod(freq_adj * max_cycles * x + phase_offset, 1)

    def pwm_extended(x, phase_offset, freq_adj, virtual_led_count, fps, bpm, create_variance, note_division=4):
        # 1. Create virtual positions and indices for the virtual LED groups.
        virtual_positions = np.linspace(0, 1, virtual_led_count)
        indices = np.arange(virtual_led_count)
        if create_variance == 0.0:
            pass
        else:
            phase_offset = phase_offset + create_variance
        
        # 2. Compute the base PWM part on the virtual scale.
        #    Here we use a modulated square wave: for each virtual position, compute
        #    (freq_adj * max_cycles * virtual_position + phase_offset) mod 1, then threshold.
        base = (np.mod(freq_adj * max_cycles * virtual_positions + phase_offset, 1) < 0.5).astype(float)
        
        # 3. Compute the odd-even pattern on the virtual groups.
        if create_variance == 0.0:
            odd_even_pattern = (indices % 2).astype(float)
        else:
            odd_even_pattern = (indices % 2 + create_variance).astype(float)
        
        
        # 4. Blend the two patterns with equal weight.
        virtual_pattern = 0.5 * base + 0.5 * odd_even_pattern
        
        # 5. Discretely map the virtual pattern onto the physical LED positions.
        #    That is, divide the physical LED array into virtual_led_count bins and assign each
        #    physical LED the value of its corresponding virtual group.
        n = len(x)
        physical_pattern = np.zeros(n)
        for i in range(n):
            virtual_index = int(i * virtual_led_count / n)
            if virtual_index >= virtual_led_count:
                virtual_index = virtual_led_count - 1
            physical_pattern[i] = virtual_pattern[virtual_index]
        
        return physical_pattern
    
    def pwm_basic(x, phase_offset, freq_adj, virtual_led_count, fps, bpm, create_variance, note_division=4):
        # Calculate beat timing
        frames_per_beat = (60 / bpm * 2) * fps
        frames_per_note = frames_per_beat / note_division
        
        # Convert phase to frames and normalize to the note division timing
        current_frame = int(phase_offset * 2 / (2 * np.pi) * fps)
        beat_position = (current_frame / frames_per_note) * (2 * np.pi)  # Scale to match the note timing
        
        # Create virtual positions
        virtual_positions = np.linspace(0, 1, virtual_led_count)
        
        # Define how many virtual LEDs should be on
        # active_leds = 2  # You can adjust this number
        active_leds = int(virtual_led_count / 2)

        # Calculate position with wrapping
        # positions = np.mod(virtual_positions + beat_position / (2 * np.pi), 1)

        if create_variance == 0.0:
            # Calculate position with wrapping
            positions = np.mod(virtual_positions + beat_position / (2 * np.pi), 1)
        else:
            # Calculate position with wrapping
            positions = np.mod(virtual_positions + beat_position + create_variance / (2 * np.pi), 1)
        
        # Sort positions and find threshold for top N values
        sorted_positions = np.sort(positions)
        threshold = sorted_positions[virtual_led_count - active_leds]
        
        # Create pattern with exactly active_leds number of ones
        base = (positions >= threshold).astype(float)
        
        virtual_pattern = base
        
        # Map to physical LEDs
        n = len(x)
        physical_pattern = np.zeros(n)
        for i in range(n):
            virtual_index = int(i * virtual_led_count / n)
            if virtual_index >= virtual_led_count:
                virtual_index = virtual_led_count - 1
            physical_pattern[i] = virtual_pattern[virtual_index]
        
        return physical_pattern

    
    def odd_even(frame, virtual_led_count, physical_led_count, fps, bpm, create_variance):
        """
        Generate a virtual odd-even pattern (length = virtual_led_count) for a given frame,
        using frame counting rather than normalized time. The pattern switches every half-beat.

        Parameters:
        frame            : The current frame index (integer).
        virtual_led_count: Total number of virtual groups (e.g., 8).
        fps              : Frames per second (e.g., 30).
        bpm              : Beats per minute.

        Returns:
        virtual_pattern  : 1D numpy array of length virtual_led_count containing 0.0 or 1.0.
        """
        # Calculate the number of frames per beat.
        frames_per_beat = fps * 60.0 / bpm  # e.g. at 30 fps and 120 bpm: 30*60/120 = 15 frames per beat
        # For double the switching rate, use half the beat length.
        # half_beat_frames = frames_per_beat / 2.0
        # overhaul
        half_beat_frames = frames_per_beat / 1.0

        # Determine the half-beat index for the current frame.
        beat_index = int(frame / half_beat_frames)
        
        # Create a virtual index array for the groups.
        indices = np.arange(virtual_led_count)
        # print(f" create variance in odd even: {create_variance}")
        
        if create_variance == 0.0:
            # If the half-beat index is even, use one pattern; if odd, flip it.
            if beat_index % 2 == 0:
                virtual_pattern = (indices % 2).astype(float)
            else:
                virtual_pattern = ((indices + 1) % 2).astype(float)
        else:
            # If the half-beat index is even, use one pattern; if odd, flip it.
            if beat_index % 2 == 0:
                virtual_pattern = ((indices + 1) % 2).astype(float)
            else:
                virtual_pattern = ((indices) % 2).astype(float)

        n = physical_led_count
        virtual_led_count = len(virtual_pattern)
        physical_pattern = np.zeros(n)
        for i in range(n):
            # Compute the bin index for the i-th physical LED.
            virtual_index = int(i * virtual_led_count / n)
            if virtual_index >= virtual_led_count:
                virtual_index = virtual_led_count - 1
            physical_pattern[i] = virtual_pattern[virtual_index]
        return physical_pattern


    def random(frame, virtual_led_count, physical_led_count, fps, bpm, create_variance, update_modifier=1):
        """
        Generates a random pattern of intensities that persists for a chosen interval.
        
        The base update interval is one quarter-beat (derived from the BPM and FPS).
        This can be modified by the update_modifier parameter:
        - update_modifier = 1  → update every quarter-beat.
        - update_modifier = 2  → update every two quarter-beats (half beat).
        - update_modifier = 8  → update every eight quarter-beats.
        
        At least 30% of the virtual LED intensity values are forced to 0.0 for contrast.
        
        Parameters:
            frame: The current frame index (integer).
            virtual_led_count: Total number of virtual LEDs.
            physical_led_count: Total number of physical LEDs.
            fps: Frames per second.
            bpm: Beats per minute.
            create_variance: Not directly used in the random pattern, kept for signature compatibility.
            update_modifier: Multiplier for the base quarter-beat interval (default 1).
        
        Returns:
            A NumPy array of length physical_led_count with the generated intensity values.
        """
        # Calculate frames per beat and derive the quarter-beat interval.
        frames_per_beat = fps * 60.0 / bpm
        quarter_beat_frames = frames_per_beat / 4.0

        # Determine the update interval in frames.
        update_interval = int(quarter_beat_frames * update_modifier)
        
        # Determine the current update index (state identifier)
        update_index = frame // update_interval

        # Create a local random generator seeded by the update_index.
        rng = np.random.default_rng(update_index)
        
        # Generate the virtual LED pattern.
        virtual_pattern = rng.random(virtual_led_count)
        
        # Force at least 30% of the virtual LED intensities to 0.0.
        num_to_zero = int(np.ceil(0.3 * virtual_led_count))
        indices_to_zero = rng.choice(virtual_led_count, num_to_zero, replace=False)
        virtual_pattern[indices_to_zero] = 0.0

        # Map the virtual pattern to the physical LEDs.
        physical_pattern = np.zeros(physical_led_count)
        for i in range(physical_led_count):
            virtual_index = int(i * virtual_led_count / physical_led_count)
            if virtual_index >= virtual_led_count:
                virtual_index = virtual_led_count - 1
            physical_pattern[i] = virtual_pattern[virtual_index]
            
        return physical_pattern

    
    def still(x, phase_offset, freq_adj, virtual_led_count, create_variance):
        """
        Returns:
            A NumPy array with the same shape as x, filled with the constant value 0.75.
        """
        return np.full_like(x, 0.75)

    # Map the decision to one of the candidate functions.
    candidate_functions = {
        "sine": sine_basis,
        "square": square_basis,
        "saw_up": saw_up_basis,
        "saw_down": saw_down_basis,
        "pwm_basic": pwm_basic,
        "pwm_extended": pwm_extended,
        "odd_even": odd_even,
        "random" : random,
        "still" : still
    }
    
    if decision not in candidate_functions:
        print("Unknown decision; defaulting to sine.")
        decision = "sine"
    candidate_fn = candidate_functions[decision]

    waveform = np.zeros((frames, effective_led_count))

    x = np.linspace(0, 1, effective_led_count)   # spatial positions (normalized)
    t = np.linspace(0, 1, frames)                # normalized time

    # Adjusted frequency based on the multiplier.
    freq_adj_local = f0 * overall_dynamic
    # print(f'freq_adj_local: {freq_adj_local}')
    for frame in range(frames):
        current_phase_offset = phase_movement * frame
        # For candidate functions that require the extra parameters:
        if candidate_fn.__name__ in ["odd_even", "random"]:
            # Get the current time from the t array.
            waveform[frame, :] = candidate_fn(frame, virtual_led_count, effective_led_count, fps, bpm, create_variance)
        elif candidate_fn.__name__ in ["pwm_basic", "pwm_extended"]:
            # t_current = t[frame]  # Ensure that t is in seconds (if t is normalized, convert to absolute time)
            waveform[frame, :] = candidate_fn(x, current_phase_offset, freq_adj_local, virtual_led_count, fps, bpm, create_variance)
        else:
            waveform[frame, :] = candidate_fn(x, current_phase_offset, freq_adj_local, virtual_led_count, create_variance)
    # composite_waveform = waveform * scale + shift
    composite_waveform = waveform
    # print(f'effective_led_count: {effective_led_count}')
    
    # ---------------------------
    # 8. Mirror the waveform (if needed) to create the full LED array.
    # ---------------------------
    if mirroring_active:
        full_waveform = np.zeros((frames, LED_COUNT))
        for frame in range(frames):
            # Calculate the middle point
            middle = (LED_COUNT + 1) // 2
            # Take the left half of composite_waveform
            left_side = composite_waveform[frame, :middle]
            # For odd numbers, we don't want to repeat the middle LED
            if LED_COUNT % 2 == 0:
                right_side = left_side[::-1]  # Reverse the left side
            else:
                right_side = left_side[:-1][::-1]  # Exclude middle LED when reversing
            full_waveform[frame, :] = np.concatenate([left_side, right_side])
        composite_waveform = full_waveform
    else:
        # If not mirroring, ensure the waveform has LED_COUNT columns
        pass

    # ---------------------------
    # 9. Convert the waveform to an RGB array via HSV conversion.
    #     For hue and saturation we blend the geo values with the PAS suggestions.
    # ---------------------------
    rgb_array = np.zeros((frames, LED_COUNT, 3))
    # print(f" Col_Hue before construcing RGB Array: {col_hue}")
    # print(f" Col_Sat before construcing RGB Array: {col_sat}")
    for frame in range(frames):
        v = composite_waveform[frame, :]
        # Blend hue and saturation: here we simply average the geo value with the mean PAS value.
        # h = ((col_hue_geo + np.mean(luminaire_dict['PASv02_allframes'][:, 4])) / 2.0) * np.ones_like(v)
        # s = ((col_sat_geo + np.mean(luminaire_dict['PASv02_allframes'][:, 5])) / 2.0) * np.ones_like(v)
        # Using hue and saturation: here we use the PAS values directly.
        # h = ((np.mean(luminaire_dict['PASv02_allframes'][:, 4])) / 1.0) * np.ones_like(v)
        # s = ((np.mean(luminaire_dict['PASv02_allframes'][:, 5])) / 1.0) * np.ones_like(v)
        c = v * (col_sat * np.ones_like(v))
        h_prime = (col_hue * np.ones_like(v)) * 6.0
        x_color = c * (1 - np.abs(np.mod(h_prime, 2) - 1))
        m = v - c
        
        rgb = np.zeros((len(v), 3))
        mask = (h_prime < 1)
        rgb[mask] = np.column_stack((c[mask], x_color[mask], np.zeros_like(x_color[mask])))
        mask = (1 <= h_prime) & (h_prime < 2)
        rgb[mask] = np.column_stack((x_color[mask], c[mask], np.zeros_like(x_color[mask])))
        mask = (2 <= h_prime) & (h_prime < 3)
        rgb[mask] = np.column_stack((np.zeros_like(x_color[mask]), c[mask], x_color[mask]))
        mask = (3 <= h_prime) & (h_prime < 4)
        rgb[mask] = np.column_stack((np.zeros_like(x_color[mask]), x_color[mask], c[mask]))
        mask = (4 <= h_prime) & (h_prime < 5)
        rgb[mask] = np.column_stack((x_color[mask], np.zeros_like(x_color[mask]), c[mask]))
        mask = (5 <= h_prime)
        rgb[mask] = np.column_stack((c[mask], np.zeros_like(x_color[mask]), x_color[mask]))
        rgb += m[:, np.newaxis]
        
        rgb_array[frame] = rgb
    
    return rgb_array

#######################

def set_color_for_segment(col_dict):

    col_hue_GEO = col_dict['col_hue_GEO']
    col_sat_GEO = col_dict['col_sat_GEO']
    col_hue_PAS = col_dict['col_hue_PAS']
    col_sat_PAS = col_dict['col_sat_PAS']

    col_hue = col_hue_PAS
    col_sat = col_sat_PAS

    return col_hue, col_sat

#######################
'''
def construct_rgb_array_for_standard_segment(luminaire_dict):
    # Constants for wave generation
    MAX_CYCLES_PER_SECOND = 4  # Maximum number of wave cycles at freq=1.0
    MAX_PHASE_CYCLES_PER_SECOND = 8  # Maximum number of phase cycles moved in one second at phase=1.0
    LED_COUNT = 33  # Number of RGB values to generate
    
    frames = luminaire_dict['frames']
    wave_type = luminaire_dict['wave_type']
    freq = luminaire_dict['freq']
    phase = luminaire_dict['phase']
    col_hue = luminaire_dict['col_hue']
    col_sat = luminaire_dict['col_sat']
    moving_direction = luminaire_dict['moving_direction']
    mirroring_active = luminaire_dict['mirroring_active']
    bpm_value = ['bpm']
    PASv02_allframes = luminaire_dict['PASv02_allframes']

    intensityPeakPASv02 = PASv02_allframes[:, 0]
    peakDensityPASv02 = PASv02_allframes[:, 1]
    intentsityInverseMinimaPASv02 = PASv02_allframes[:, 3]
    colHuePASv02 = PASv02_allframes[:, 4]
    colSatPeakPASv02 = PASv02_allframes[:, 5]

    # Initialize output array (frames, LED_count, RGB)
    rgb_array = np.zeros((frames, LED_COUNT, 3))
    
    # Calculate mean phase for movement
    mean_phase = np.mean(phase)
    
    if not mirroring_active:
        LED_COUNT = LED_COUNT
    else:
        LED_COUNT = LED_COUNT // 2 + 1
        # MAX_PHASE_CYCLES_PER_SECOND = MAX_PHASE_CYCLES_PER_SECOND // 2
        MAX_CYCLES_PER_SECOND = MAX_CYCLES_PER_SECOND // 2

    # Generate position array for LEDs (0 to 1)
    x = np.linspace(0, 1, LED_COUNT)

    # Generate time array for frames
    t = np.linspace(0, 1, frames)
    
    # Calculate phase movement per frame
    phase_movement = (mean_phase * MAX_PHASE_CYCLES_PER_SECOND / frames) * (1 if moving_direction < 0.5 else -1)
    
    # Generate base wave pattern based on wave type
    for frame in range(frames):
        # Calculate phase offset for current frame
        current_phase = phase_movement * frame
        
        # Generate wave pattern based on type
        if wave_type == 'sine':
            values = 0.5 + 0.5 * np.sin(2 * np.pi * (freq * MAX_CYCLES_PER_SECOND * x + current_phase))
        elif wave_type == 'saw_up':
            values = np.mod(freq * MAX_CYCLES_PER_SECOND * x + current_phase, 1)
        elif wave_type == 'saw_down':
            values = 1 - np.mod(freq * MAX_CYCLES_PER_SECOND * x + current_phase, 1)
        elif wave_type == 'square':
            values = (np.mod(freq * MAX_CYCLES_PER_SECOND * x + current_phase, 1) < 0.5).astype(float)
        elif wave_type == 'linear':
            values = x

        # print("All values:", values)
        # print("Min value:", np.min(values))
        # print("Max value:", np.max(values))
        
        # Mirror if active
        if mirroring_active:
            values = np.concatenate([values[:-1], values[::-1]])
        
        # Convert HSV to RGB
        h = col_hue * np.ones_like(values)
        s = col_sat * np.ones_like(values)
        v = values
        
        # HSV to RGB conversion
        c = v * s
        h_prime = h * 6
        x_color = c * (1 - np.abs(np.mod(h_prime, 2) - 1))
        m = v - c
        
        rgb = np.zeros((len(values), 3))
        
        # RGB assignment based on hue
        mask = (h_prime < 1)
        rgb[mask] = np.column_stack([c[mask], x_color[mask], np.zeros_like(x_color[mask])])
        mask = (1 <= h_prime) & (h_prime < 2)
        rgb[mask] = np.column_stack([x_color[mask], c[mask], np.zeros_like(x_color[mask])])
        mask = (2 <= h_prime) & (h_prime < 3)
        rgb[mask] = np.column_stack([np.zeros_like(x_color[mask]), c[mask], x_color[mask]])
        mask = (3 <= h_prime) & (h_prime < 4)
        rgb[mask] = np.column_stack([np.zeros_like(x_color[mask]), x_color[mask], c[mask]])
        mask = (4 <= h_prime) & (h_prime < 5)
        rgb[mask] = np.column_stack([x_color[mask], np.zeros_like(x_color[mask]), c[mask]])
        mask = (5 <= h_prime)
        rgb[mask] = np.column_stack([c[mask], np.zeros_like(x_color[mask]), x_color[mask]])
        
        # Add value offset
        rgb += m[:, np.newaxis]
        
        rgb_array[frame] = rgb


    # x = np.linspace(0, 1, LED_COUNT)  # This creates evenly spaced points
    # x = x * 0.5
    # print("x:", x)
    # print("freq:", freq)
    # print("MAX_CYCLES_PER_SECOND:", MAX_CYCLES_PER_SECOND)
    # print("current_phase:", current_phase)

    # print(f"Type of values: {type(values)}, Shape: {getattr(values, 'shape', 'No shape attribute')}")
    # print(f"wave txpe: {wave_type}")
    
    return rgb_array
'''
#######################

def modify_rgb_array_with_highlight_segment(luminaire_dict, config):
    """
    Modifies an RGB array to add a highlight segment.

    Args:
        luminaire_dict: A dictionary containing parameters for the highlight.

    Returns:
        A modified RGB array with the highlight applied.
    """

    # Constants
    LED_COUNT = config["led_count"]

    # Extract parameters from the dictionary
    frames = luminaire_dict['frames']
    wave_type = luminaire_dict['wave_type']
    rgb_array = luminaire_dict['rgb_standard_array']
    highlights = luminaire_dict['highlights']
    highlight_allframes = luminaire_dict['highlight_allframes']
    amplitude_allframes = highlight_allframes[:, 5]

    # Iterate through each frame
    for frame in range(frames):
        amplitude = amplitude_allframes[frame]
        # print(f"frame: {frame}")

        if highlights:
            # Apply highlight for 'plateau' wave type
            if wave_type == 'plateau':
                if amplitude > 0.5:
                    # Blend towards white if amplitude is between 0.5 and 0.7
                    if amplitude <= 0.7:
                        blend_factor = (amplitude - 0.5) / 0.2  # Scale to 0-1 range
                        white_rgb = np.array([1.0, 1.0, 1.0])
                        current_rgb = rgb_array[frame]
                        blended_rgb = (1 - blend_factor) * current_rgb + blend_factor * white_rgb
                        rgb_array[frame] = blended_rgb
                    # Full white if amplitude is greater than 0.7
                    else:
                        rgb_array[frame] = np.ones((LED_COUNT, 3))  # All LEDs white
            else:
                rgb_array[frame] = rgb_array[frame] # No change

    # print('Highlight appllied with wave type:', wave_type)
    return rgb_array

#######################

def get_highlight_time_positions_out_of_segment(highlight_allframes, frames, config):
    # Constants
    LED_COUNT = config["led_count"]
    fps = config["fps"]

    # Extract parameters from the dictionary
    amplitude_allframes = highlight_allframes[:, 5]
    highlight_timings = {}

    # Variables to track highlight status
    highlight_start_frame = None
    highlight_index = 0

    # Iterate through each frame
    for frame in range(frames):
        amplitude = amplitude_allframes[frame]
        # print(f"frame: {frame}")

        if amplitude > 0.6:
            # If we're not already tracking a highlight, mark this as the start frame
            if highlight_start_frame is None:
                highlight_start_frame = frame
        else:
            # If we were tracking a highlight and it just ended
            if highlight_start_frame is not None:
                # Calculate timing information
                start_time = highlight_start_frame / fps  # Starting time in seconds
                duration = (frame - highlight_start_frame) / fps  # Duration in seconds
                
                # Store in the highlight_timings dictionary in JSON-compatible format
                highlight_timings[str(highlight_index)] = {
                    "start_time": start_time,
                    "duration": duration
                }
                
                # Reset tracking and increment the index
                highlight_start_frame = None
                highlight_index += 1

    # Handle the case where a highlight is still ongoing at the end of frames
    if highlight_start_frame is not None:
        start_time = highlight_start_frame / fps
        duration = (frames - highlight_start_frame) / fps
        
        highlight_timings[str(highlight_index)] = {
            "start_time": start_time,
            "duration": duration
        }

    # print('Highlight appllied with wave type:', wave_type)
    return highlight_timings

#######################

def write_variables_to_json(config, luminaire_dict, array_descision_dict, song_part_name, lx_highlights_timings, relative_path_to_store_the_JSONs):
    # Meta information variables
    LED_COUNT = config["led_count"]
    fps = config["fps"]
    virtual_led_count = config["virtual_led_count"]
    max_cycles = config["max_cycles_per_second"]
    effective_led_count = LED_COUNT

    frames = luminaire_dict['frames']
    bpm = luminaire_dict.get('bpm', 120)  # default if not provided
    current_lx_number = str(luminaire_dict['current_lx_number'])
    PASv02_to_lx = str(luminaire_dict['PASv02_to_lx'])
    PASv02_to_lx = 'PASv02_number_' + PASv02_to_lx + 'to_' + current_lx_number

    # Adjustable parameters
    mirroring_active = luminaire_dict["mirroring_active"]
    moving_direction = luminaire_dict["moving_direction"]
    highlights = luminaire_dict["highlights"]

    decision = array_descision_dict['decision']
    overall_dynamic = array_descision_dict['overall_dynamic']
    f0 = array_descision_dict['f0']
    phase_movement = array_descision_dict['phase_movement']
    col_hue = array_descision_dict['col_hue']
    col_sat = array_descision_dict['col_sat']
    create_variance = array_descision_dict['create_variance']
    lead_group_str = array_descision_dict['lead_group_str']
    lx_position = array_descision_dict['lx_position']
    lx_movement = array_descision_dict['lx_movement']

    # Create a hierarchical JSON structure, ensuring all numbers are standard Python floats
    data = {
        "meta_information": {
            "LED_COUNT": int(LED_COUNT),
            "fps": int(fps),
            "virtual_led_count": int(virtual_led_count),
            "max_cycles": float(max_cycles),
            "effective_led_count": int(effective_led_count),
            "frames": int(frames),
            "bpm": float(bpm),
            "current_lx_number": current_lx_number,
            "song_part_name": song_part_name,
            "PASv02_to_lx": PASv02_to_lx
        },
        "highlights_suggestions": {
            "highlights_suggestions": lx_highlights_timings
        },
        "adjustable_parameters": {
            "decision": {
                "value": decision,
                "manipulator": "dropdown",
                "options": ["sine", "square", "saw_up", "saw_down", "pwm_basic", "pwm_extended", "odd_even", "random", "still"] 
            },
            "lead_group_str": {
                "value": lead_group_str,
                "manipulator": "dropdown",
                "options": ["Lead_Grouping_bypass", "Lead_Grouping_Master_LX1", "Lead_Grouping_Master_LX2", "Lead_Grouping_Master_LX3"] 
            },
            "mirroring_active": {
                "value": float(mirroring_active),
                "manipulator": "toggle",
                "options": [0.0, 1.0] 
            },
            "moving_direction": {
                "value": float(moving_direction),
                "manipulator": "toggle",
                "options": [0.0, 1.0] 
            },
            "overall_dynamic": {
                "value": float(overall_dynamic),
                "manipulator": "slider",
                "range": [0.0, 10.0]
            },
            "f0": {
                "value": float(f0),
                "manipulator": "slider",
                "range": [0.0, 10.0]
            },
            "phase_movement": {
                "value": float(phase_movement),
                "manipulator": "slider",
                "range": [0.0, 10.0]
            },
            "lx_position": {
                "value": lx_position,
                "manipulator": "dropdown",
                "options": ['Pos01', 'Pos02', 'Pos03', 'Pos04']
            },
            "lx_movement": {
                "value": float(lx_movement),
                "manipulator": "slider",
                "range": [0.0, 1.0]
            },         
            "col_hue": {
                "value": float(col_hue),
                "manipulator": "slider",
                "range": [0.0, 1.0]
            },
            "col_sat": {
                "value": float(col_sat),
                "manipulator": "slider",
                "range": [0.0, 1.0]
            },
            "create_variance": {
                "value": float(create_variance),
                "manipulator": "toggle",
                "options": [0.0, 1.0] 
            },
            "highlights": {
                "value": float(highlights),
                "manipulator": "toggle",
                "options": [0.0, 1.0] 
            }
        }
    }

    directory = os.path.dirname(relative_path_to_store_the_JSONs)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Construct the full file path for the JSON file
    json_file_name = f"{song_part_name}_{current_lx_number}_lighting_standard_config.json"
    json_file_path = os.path.join(relative_path_to_store_the_JSONs, json_file_name)
    
    
    # Write the data to the JSON file
    with open(json_file_path, "w") as outfile:
        json.dump(data, outfile, indent=4)

    return data

#######################

# --- Function to Send Feature Updates ---
def send_feature_update(message_to_send):
    """Sends a FEATURE_UPDATED message to the server."""

    message = {
        "type": "FEATURE_UPDATED",
        "message_to_send": message_to_send,
    }
    
    sio.emit('message', json.dumps(message), namespace='/')
    print ("Sent FEATURE_UPDATED from TD to server.py")

#######################

def allframesofasegment_to_pantilt_activity(al_GeoApproach_allframes):
    pan_activity = al_GeoApproach_allframes[:, 0]
    pan_activity_delta = np.max(pan_activity) - np.min(pan_activity)
    tilt_activity = al_GeoApproach_allframes[:, 1]
    tilt_activity_delta = np.max(tilt_activity) - np.min(tilt_activity)

    if pan_activity_delta > 0.15:
        pan_activity = pan_activity 
    else:
        pan_activity[:] = np.mean(pan_activity)

    if tilt_activity_delta > 0.15:  
        tilt_activity = tilt_activity
    else:
        tilt_activity[:] = np.mean(tilt_activity)

    return pan_activity, tilt_activity

#######################

def allframesofasegment_to_phase(al_GeoApproach_allframes):
    phase = al_GeoApproach_allframes[:, 7]
    phase_delta = np.max(phase) - np.min(phase)

    if phase_delta > 0.15:
        phase = phase
    else:
        phase[:] = np.mean(phase)

    return phase

#######################

def allframesofasegment_to_offset(al_GeoApproach_allframes):
    offset = al_GeoApproach_allframes[:, 6]
    offset_delta = np.max(offset) - np.min(offset)

    if offset_delta > 0.15:
        offset = offset
    else:
        offset[:] = np.mean(offset)

    return offset

#######################

def allframesofasegment_to_freq(al_GeoApproach_allframes):
    freq = al_GeoApproach_allframes[:, 4]
    freq_delta = np.max(freq) - np.min(freq)
    if freq_delta > 0.15:
        freq = np.max(freq)
    else:
        freq = np.mean(freq)

    freq = float(freq)
    return freq

#######################

def allframesofasegment_to_colhue_colsat(al_GeoApproach_allframes):
    colhue = al_GeoApproach_allframes[:, 8]
    colsat = al_GeoApproach_allframes[:, 9]

    colhue_delta = np.max(colhue) - np.min(colhue)
    colsat_delta = np.max(colsat) - np.min(colsat)

    if colhue_delta > 0.15:
        colhue = np.max(colhue)
    else:        
        colhue = np.mean(colhue)

    if colsat_delta > 0.15:
        colsat = np.max(colsat)
        # Multiply by 1.5, then clamp between 0.0 and 1.0
        colsat = np.clip(colsat * 2.0, a_min=0., a_max=1.)
        print("ColSat at Delta higher 0.15", colsat)
    else:
        colsat = np.mean(colsat)
        colsat = np.clip(colsat * 2.0, a_min=0., a_max=1.)
        print("ColSat at Delta lower 0.15", colsat)



    colhue = float(colhue)
    colsat = float(colsat)

    return colhue, colsat

#######################

def allframesofasegment_to_wavetype(al_GeoApproach_allframes):
    wave_type_a = al_GeoApproach_allframes[:, 2]
    wave_type_a = np.mean(wave_type_a)
    wave_type_b = al_GeoApproach_allframes[:, 3]
    wave_type_b_delta = np.max(wave_type_b) - np.min(wave_type_b)
    wave_type_b_max = np.max(wave_type_b)
    if wave_type_b_delta > 0.15:
        # wave_type_b_active = 1
        if 0.25 <= wave_type_b_max < 0.5:
            wave_type_string = "plateau"
        elif 0.5 <= wave_type_b_max < 0.75:
            wave_type_string = "gaussian_single"
        elif 0.75 <= wave_type_b_max <= 1.0:
            wave_type_string = "gaussian_double"
        else:
            wave_type_string = "none"
    else:
        # wave_type_b_active = 0
        if 0.0 <= wave_type_a < 0.2:
            wave_type_string = "sine"
        elif 0.2 <= wave_type_a < 0.4:
            wave_type_string = "saw_up"
        elif 0.4 <= wave_type_a < 0.6:
            wave_type_string = "saw_down"
        elif 0.6 <= wave_type_a < 0.8:
            wave_type_string = "square"
        elif 0.8 <= wave_type_a <= 1.0:
            wave_type_string = "linear"
        else:
            wave_type_string = "none"

    wave_type = wave_type_string
    return wave_type

#######################

def RGB_array_to_WashZFXPro(RGB_array, pan_val=0.5, tilt_val=0.5):
    """
    Converts an RGB array into a DMX channel configuration for Clay Paky Alpha Beam 700.

    Args:
    RGB_array (numpy.ndarray): Input array of shape (num_luminaires, 3) with RGB values between 0.0 and 1.0.

    Returns:
    numpy.ndarray: DMX channel values in shape (num_luminaires * 22,)
    """

    num_luminaires = RGB_array.shape[0]
    
    # Initialize DMX array (num_luminaires, 20)
    DMX_array = np.zeros((num_luminaires, 20))

    # RGB Mode Conversion
    DMX_array[:, 4] = RGB_array[:, 0]  # Cyan
    DMX_array[:, 5] = RGB_array[:, 1]  # Magenta
    DMX_array[:, 6] = RGB_array[:, 2]  # Yellow

    # Fixed values
    DMX_array[:, 8] = 1.0  # Stop/Strobe at 100%
    DMX_array[:, 9] = 1.0  # Dimmer at 100%
    DMX_array[:, 0] = pan_val  # Pan Coarse at 50%
    DMX_array[:, 1] = pan_val  # Pan Coarse at 50%
    DMX_array[:, 2] = tilt_val  # Tilt Coarse at 50%
    DMX_array[:, 3] = tilt_val  # Tilt Coarse at 50%

    # Flatten the array to match DMX sequential format
    return DMX_array.flatten()


#######################

def RGB_array_to_RobinCuete(RGB_array, pan_array, tilt_array, virtual_led_count):

    gobo_value = op('Gobo_Cuete')[0,0]
    focus_value = op('Focus_Cuete')[0,0]

    num_luminaires = RGB_array.shape[0]
    
    # Initialize DMX array (num_luminaires, 20)
    DMX_array = np.zeros((num_luminaires, 29))

    DMX_array[:, 17] = gobo_value  # Gobo
    DMX_array[:, 24] = focus_value  # Focus

    # RGB Mode Conversion
    DMX_array[:, 11] = 1 - RGB_array[:, 0]  # Cyan
    DMX_array[:, 12] = 1 - RGB_array[:, 1]  # Magenta
    DMX_array[:, 13] = 1 - RGB_array[:, 2]  # Yellow

    DMX_array[:, 9] = op('Color_Wheel_Cuete')[0,0]  # Color Wheel 

    HSV_array = rgb_to_hsv(RGB_array)

    # Fixed values
    DMX_array[:, 26] = 1.0  # Shutter at 100%

    DMX_array[:, 27] = HSV_array[:,2]  # Dimmer MSB at 100%
    DMX_array[:, 28] = 1.0  # Dimmer LSB at 100%

    for i in range(num_luminaires):
        DMX_array[i, 0] = pan_array[i]  # Pan Coarse
        DMX_array[i, 1] = pan_array[i]  # Pan Fine 
        DMX_array[i, 2] = tilt_array[i]  # Tilt Coarse 
        DMX_array[i, 3] = tilt_array[i]  # Tilt Fine 

    # DMX_array[:, 0] = pan_val  # Pan Coarse at 50%
    # DMX_array[:, 1] = pan_val  # Pan Fine at 50%

    # DMX_array[:, 2] = tilt_val  # Tilt Coarse at 50%
    # DMX_array[:, 3] = tilt_val  # Tilt Fine at 50%

    # Flatten the array to match DMX sequential format
    return DMX_array.flatten()

#######################

def RGB_array_to_AlphaBeam(RGB_array, pan_array, tilt_array, virtual_led_count):

    gobo_value = op('Gobo_Cuete')[0,0]
    focus_value = op('Focus_Cuete')[0,0]

    num_luminaires = RGB_array.shape[0]
    
    # Initialize DMX array (num_luminaires, 20)
    DMX_array = np.zeros((num_luminaires, 22))

    DMX_array[:, 9] = gobo_value  # Gobo
    DMX_array[:, 14] = focus_value  # Focus

    # RGB Mode Conversion
    DMX_array[:, 0] = 1 - RGB_array[:, 0]  # Cyan
    DMX_array[:, 1] = 1 - RGB_array[:, 1]  # Magenta
    DMX_array[:, 2] = 1 - RGB_array[:, 2]  # Yellow

    DMX_array[:, 3] = op('Color_Wheel_Cuete')[0,0]  # Color Wheel 

    HSV_array = rgb_to_hsv(RGB_array)

    # Fixed values
    DMX_array[:, 4] = 1.0  # Stop / Strobe at 100%
    DMX_array[:, 7] = 1.0  # Stop / Strobe at 100%

    # Dimmer is using 
    DMX_array[:, 5] = HSV_array[:,2]  # Dimmer MSB at 100%
    DMX_array[:, 6] = 0.0  # Dimmer LSB at 0%

    for i in range(num_luminaires):
        DMX_array[i, 15] = pan_array[i]  # Pan Coarse
        DMX_array[i, 16] = pan_array[i]  # Pan Fine 
        DMX_array[i, 17] = tilt_array[i]  # Tilt Coarse 
        DMX_array[i, 18] = tilt_array[i]  # Tilt Fine 

    # DMX_array[:, 0] = pan_val  # Pan Coarse at 50%
    # DMX_array[:, 1] = pan_val  # Pan Fine at 50%

    # DMX_array[:, 2] = tilt_val  # Tilt Coarse at 50%
    # DMX_array[:, 3] = tilt_val  # Tilt Fine at 50%

    # Flatten the array to match DMX sequential format
    return DMX_array.flatten()

#######################

def RGB_array_to_RobeLedBeam150(RGB_array, pan_array, tilt_array, zoom_val=0.999):
    """
    Converts an RGB array into a DMX channel configuration for Robin Cuete.

    Args:
    RGB_array (numpy.ndarray): Input array of shape (num_luminaires, 3) with RGB values between 0.0 and 1.0.

    Returns:
    numpy.ndarray: DMX channel values in shape (num_luminaires * 22,)
    """

    LED_Beam_RGB_Correction_Factor_R = op('LED_Beam_RGB_Correction_Factor_R')[0,0]
    LED_Beam_RGB_Correction_Factor_G = op('LED_Beam_RGB_Correction_Factor_G')[0,0]
    LED_Beam_RGB_Correction_Factor_B = op('LED_Beam_RGB_Correction_Factor_B')[0,0]
    LED_Beam_RGB_Correction_Factor_DIM = op('LED_Beam_RGB_Correction_Factor_DIM')[0,0]

    num_luminaires = RGB_array.shape[0]
    
    # Initialize DMX array (num_luminaires, 20)
    DMX_array = np.zeros((num_luminaires, 16))

    # RGB Mode Conversion
    DMX_array[:, 7] = RGB_array[:, 0] * LED_Beam_RGB_Correction_Factor_R
    DMX_array[:, 8] = RGB_array[:, 1] * LED_Beam_RGB_Correction_Factor_G
    DMX_array[:, 9] = RGB_array[:, 2] * LED_Beam_RGB_Correction_Factor_B
    # DMX_array[:, 7] = 1.0
    # DMX_array[:, 8] = 1.0
    # DMX_array[:, 9] = 1.0
    # DMX_array[:, 10] = 1.0 # White

    HSV_array = rgb_to_hsv(RGB_array)

    # Fixed values
    DMX_array[:, 12] = 0.176470  # Colormix
    DMX_array[:, 14] = 1.0  # Shutter at 100%
    DMX_array[:, 15] = 1.0 * LED_Beam_RGB_Correction_Factor_DIM # Dimmer 


    # DMX_array[:, 27] = HSV_array[:,2]  # Dimmer MSB at 100%
    # DMX_array[:, 28] = 1.0  # Dimmer LSB at 100%

    DMX_array[:, 13] = zoom_val  # Zoom at 50% 

    for i in range(num_luminaires):
        DMX_array[i, 0] = pan_array[i]  # Pan Coarse
        DMX_array[i, 1] = pan_array[i]  # Pan Fine 
        DMX_array[i, 2] = tilt_array[i]  # Tilt Coarse 
        DMX_array[i, 3] = tilt_array[i]  # Tilt Fine 

    # Flatten the array to match DMX sequential format
    return DMX_array.flatten()

def RGB_array_to_ColorFusion(RGB_array):

    num_luminaires = RGB_array.shape[0]
    DMX_array = np.zeros((num_luminaires, 4))
    fixed_dim = 1.0


    DMX_array[:, 0] = RGB_array[:, 0] * fixed_dim
    DMX_array[:, 1] = RGB_array[:, 1] * fixed_dim
    DMX_array[:, 2] = RGB_array[:, 2] * fixed_dim
    DMX_array[:, 3] = 1.0  # Strobe
    return DMX_array.flatten() 

#######################

def bound(x, bl, bu):
    # return bounded value clipped between bl and bu
    y = min(max(x, bl), bu)
    return y

#######################

def dump_numpy_to_pickle(array, relative_filepath):
  """Dumps a NumPy array to a pickle file using a relative path.

  Args:
    array: The NumPy array to be saved.
    relative_filepath: The relative path to the pickle file (e.g., "data/my_array.pkl").
  """

  # Create the directory if it doesn't exist
  directory = os.path.dirname(relative_filepath)
  if directory and not os.path.exists(directory):
    os.makedirs(directory)

  # Get the absolute path
  absolute_filepath = os.path.abspath(relative_filepath)

  # Dump the array to the pickle file
  with open(absolute_filepath, 'wb') as f:
    pickle.dump(array, f)

  # print(f"NumPy array saved to: {absolute_filepath}")
  
    """
    # Save to a file named 'my_array.pkl' in a subdirectory named 'data'
    dump_numpy_to_pickle(my_array, "data/my_array.pkl")
    """

#######################

def rgb_to_hsv(rgb_array):
    # Ensure RGB values are between 0 and 1
    rgb_array = np.clip(rgb_array, 0, 1)
    
    r, g, b = rgb_array[:, 0], rgb_array[:, 1], rgb_array[:, 2]
    
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc
    
    # Initialize HSV arrays
    h = np.zeros_like(r)
    s = np.zeros_like(r)
    v = maxc
    
    # Calculate Hue
    mask = delta != 0
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    
    rc[mask] = (maxc[mask] - r[mask]) / delta[mask]
    gc[mask] = (maxc[mask] - g[mask]) / delta[mask]
    bc[mask] = (maxc[mask] - b[mask]) / delta[mask]
    
    h[maxc == r] = bc[maxc == r] - gc[maxc == r]
    h[maxc == g] = 2.0 + rc[maxc == g] - bc[maxc == g]
    h[maxc == b] = 4.0 + gc[maxc == b] - rc[maxc == b]
    
    h = (h / 6.0) % 1.0
    
    # Calculate Saturation
    s[maxc != 0] = delta[maxc != 0] / maxc[maxc != 0]
    
    return np.column_stack((h, s, v))

#######################

def load_numpy_from_pickle(filepath):
  """Loads a NumPy array from a pickle file.

  Args:
    filepath: The path to the pickle file.

  Returns:
    The loaded NumPy array.
  """
  with open(filepath, 'rb') as f:
    array = pickle.load(f)

    """
    Example usage (assuming you've already saved the array as shown above):
    loaded_array = load_numpy_from_pickle("data/my_array.pkl")
    print(loaded_array) 
    """
  return array