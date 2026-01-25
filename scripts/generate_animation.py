"""
Generate animated visualization of trains moving through the London Underground network.
Uses VehicleJourneys timetable data to show realistic train movements.
"""
import pyarrow.feather as feather
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from pathlib import Path
from datetime import datetime, timedelta
import re

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed_data'
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

# TfL line colors
LINE_COLORS = {
    'BAK': '#B36305', 'CEN': '#E32017', 'CIR': '#FFD300', 'DIS': '#00782A',
    'DLR': '#00A4A7', 'ELZ': '#6950A1', 'HAM': '#F3A9BB', 'JUB': '#A0A5A9',
    'MET': '#9B0056', 'NTN': '#000000', 'PIC': '#003688', 'TRM': '#84B817',
    'VIC': '#0098D4', 'WAC': '#95CDBA',
}

LINE_NAMES = {
    'BAK': 'Bakerloo', 'CEN': 'Central', 'CIR': 'Circle', 'DIS': 'District',
    'DLR': 'DLR', 'ELZ': 'Elizabeth', 'HAM': 'Hammersmith & City', 'JUB': 'Jubilee',
    'MET': 'Metropolitan', 'NTN': 'Northern', 'PIC': 'Piccadilly', 'TRM': 'Trams',
    'VIC': 'Victoria', 'WAC': 'Waterloo & City',
}


def parse_duration(duration_str):
    """Parse ISO 8601 duration string (e.g., 'PT1M', 'PT2M30S') to seconds."""
    if pd.isna(duration_str) or duration_str is None:
        return 0
    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', str(duration_str))
    if match:
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        return hours * 3600 + minutes * 60 + seconds
    return 0


def parse_time(time_str):
    """Parse time string to seconds since midnight."""
    if pd.isna(time_str):
        return None
    parts = str(time_str).split(':')
    if len(parts) >= 2:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2]) if len(parts) > 2 else 0
        return hours * 3600 + minutes * 60 + seconds
    return None


def load_stops():
    """Load station coordinates."""
    stops = feather.read_feather(DATA_DIR / 'ALL-StopPoints.feather')
    stops['Easting'] = pd.to_numeric(stops['Place_Location_Easting'])
    stops['Northing'] = pd.to_numeric(stops['Place_Location_Northing'])
    return stops.set_index('AtcoCode')[['Easting', 'Northing', 'Descriptor_CommonName']].to_dict('index')


def load_line_data(line_code):
    """Load all data for a single line."""
    prefix = DATA_DIR / f'{line_code}-'

    data = {
        'journeys': feather.read_feather(f'{prefix}VehicleJourneys.feather'),
        'patterns': feather.read_feather(f'{prefix}JourneyPatterns.feather'),
        'timing_links': feather.read_feather(f'{prefix}JourneyPatternTimingLinks.feather'),
        'route_links': feather.read_feather(f'{prefix}RouteLinks.feather'),
    }
    return data


def build_journey_schedules(line_data, stop_coords):
    """
    Build a list of train schedules: each schedule is a list of
    (time_seconds, station_id, x, y) tuples representing when the train is at each station.
    """
    journeys = line_data['journeys']
    patterns = line_data['patterns']
    timing_links = line_data['timing_links']

    # Create lookup from JourneyPattern to JourneyPatternSectionRefs
    pattern_to_sections = patterns.set_index('JourneyPattern')['JourneyPatternSectionRefs'].to_dict()

    # Group timing links by section
    section_links = timing_links.groupby('JourneyPatternSections')

    schedules = []

    for _, journey in journeys.iterrows():
        jp_ref = journey['JourneyPatternRef']
        departure_time = parse_time(journey['DepartureTime'])

        if departure_time is None or jp_ref not in pattern_to_sections:
            continue

        section_ref = pattern_to_sections[jp_ref]

        if section_ref not in section_links.groups:
            continue

        links = section_links.get_group(section_ref).sort_values('From_SequenceNumber')

        schedule = []
        current_time = departure_time

        for _, link in links.iterrows():
            from_stop = link['From_StopPointRef']
            to_stop = link['To_StopPointRef']
            run_time = parse_duration(link['RunTime'])
            wait_time = parse_duration(link['WaitTime'])

            # Add departure from current station
            if from_stop in stop_coords:
                schedule.append({
                    'time': current_time,
                    'station': from_stop,
                    'x': stop_coords[from_stop]['Easting'],
                    'y': stop_coords[from_stop]['Northing'],
                })

            # Move to next station
            current_time += run_time + wait_time

        # Add final station
        if links.shape[0] > 0:
            last_link = links.iloc[-1]
            to_stop = last_link['To_StopPointRef']
            if to_stop in stop_coords:
                schedule.append({
                    'time': current_time,
                    'station': to_stop,
                    'x': stop_coords[to_stop]['Easting'],
                    'y': stop_coords[to_stop]['Northing'],
                })

        if len(schedule) >= 2:
            schedules.append(schedule)

    return schedules


def interpolate_position(schedule, current_time):
    """
    Given a train schedule and current time, interpolate the train's position.
    Returns (x, y) or None if train is not running at this time.
    """
    if not schedule:
        return None

    first_time = schedule[0]['time']
    last_time = schedule[-1]['time']

    if current_time < first_time or current_time > last_time:
        return None

    # Find the segment we're in
    for i in range(len(schedule) - 1):
        t1, t2 = schedule[i]['time'], schedule[i+1]['time']
        if t1 <= current_time <= t2:
            # Linear interpolation
            if t2 == t1:
                ratio = 0
            else:
                ratio = (current_time - t1) / (t2 - t1)

            x = schedule[i]['x'] + ratio * (schedule[i+1]['x'] - schedule[i]['x'])
            y = schedule[i]['y'] + ratio * (schedule[i+1]['y'] - schedule[i]['y'])
            return (x, y)

    return None


def create_animation(lines_to_animate=['BAK', 'VIC', 'CEN', 'NTN'],
                     start_time_str='08:00:00',
                     duration_minutes=60,
                     fps=10,
                     speed_multiplier=60):
    """
    Create animation of trains moving through the network.

    Args:
        lines_to_animate: List of line codes to include
        start_time_str: Start time (HH:MM:SS)
        duration_minutes: Real-world duration to animate
        fps: Frames per second in output
        speed_multiplier: How much faster than real-time (60 = 1 min real = 1 sec animation)
    """
    print("Loading station coordinates...")
    stop_coords = load_stops()

    # Load route links for background
    all_edges = {}
    for line_code in LINE_COLORS.keys():
        path = DATA_DIR / f'{line_code}-RouteLinks.feather'
        if path.exists():
            links = feather.read_feather(path)
            all_edges[line_code] = links[['From_StopPointRef', 'To_StopPointRef']].drop_duplicates()

    # Load schedules for animated lines
    print("Building train schedules...")
    all_schedules = {}
    for line_code in lines_to_animate:
        try:
            line_data = load_line_data(line_code)
            schedules = build_journey_schedules(line_data, stop_coords)
            all_schedules[line_code] = schedules
            print(f"  {LINE_NAMES.get(line_code, line_code)}: {len(schedules)} journeys")
        except Exception as e:
            print(f"  {line_code}: Error loading - {e}")

    # Time setup
    start_time = parse_time(start_time_str)
    end_time = start_time + duration_minutes * 60

    # Calculate frames
    animation_duration_sec = duration_minutes * 60 / speed_multiplier
    total_frames = int(animation_duration_sec * fps)

    print(f"\nAnimation: {duration_minutes} min real-time -> {animation_duration_sec:.1f}s animation")
    print(f"Frames: {total_frames} at {fps} fps")

    # Set up figure
    fig, ax = plt.subplots(figsize=(14, 16))

    # Calculate bounds
    all_x = [stop_coords[s]['Easting'] for s in stop_coords]
    all_y = [stop_coords[s]['Northing'] for s in stop_coords]
    x_margin = (max(all_x) - min(all_x)) * 0.05
    y_margin = (max(all_y) - min(all_y)) * 0.05

    ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
    ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw static background (all lines, dimmed)
    for line_code, edges in all_edges.items():
        color = LINE_COLORS.get(line_code, '#999999')
        alpha = 0.8 if line_code in lines_to_animate else 0.2
        for _, edge in edges.iterrows():
            from_s, to_s = edge['From_StopPointRef'], edge['To_StopPointRef']
            if from_s in stop_coords and to_s in stop_coords:
                x = [stop_coords[from_s]['Easting'], stop_coords[to_s]['Easting']]
                y = [stop_coords[from_s]['Northing'], stop_coords[to_s]['Northing']]
                ax.plot(x, y, color=color, linewidth=2 if line_code in lines_to_animate else 1,
                       alpha=alpha, solid_capstyle='round', zorder=1)

    # Draw stations
    station_x = [stop_coords[s]['Easting'] for s in stop_coords]
    station_y = [stop_coords[s]['Northing'] for s in stop_coords]
    ax.scatter(station_x, station_y, c='white', s=8, edgecolors='#666666',
               linewidth=0.3, zorder=5)

    # Create train scatter plots for each line
    train_scatters = {}
    for line_code in lines_to_animate:
        scatter = ax.scatter([], [], c=LINE_COLORS.get(line_code, '#000000'),
                            s=80, edgecolors='white', linewidth=1, zorder=10,
                            label=LINE_NAMES.get(line_code, line_code))
        train_scatters[line_code] = scatter

    # Time display
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=14,
                        verticalalignment='top', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Legend
    legend_elements = [Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=LINE_COLORS.get(code, '#000'),
                              markersize=10, label=LINE_NAMES.get(code, code))
                       for code in lines_to_animate]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    ax.set_title('London Underground Train Movements', fontsize=14, fontweight='bold')

    def init():
        for scatter in train_scatters.values():
            scatter.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        return list(train_scatters.values()) + [time_text]

    def animate(frame):
        # Calculate current time
        progress = frame / total_frames
        current_time = start_time + progress * (end_time - start_time)

        # Update time display
        hours = int(current_time // 3600) % 24
        minutes = int((current_time % 3600) // 60)
        time_text.set_text(f'{hours:02d}:{minutes:02d}')

        # Update train positions
        for line_code, scatter in train_scatters.items():
            if line_code not in all_schedules:
                continue

            positions = []
            for schedule in all_schedules[line_code]:
                pos = interpolate_position(schedule, current_time)
                if pos:
                    positions.append(pos)

            if positions:
                scatter.set_offsets(positions)
            else:
                scatter.set_offsets(np.empty((0, 2)))

        return list(train_scatters.values()) + [time_text]

    print("\nGenerating animation...")
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=total_frames, interval=1000/fps, blit=True)

    # Save as GIF
    output_path = OUTPUT_DIR / 'train_animation.gif'
    print(f"Saving to {output_path}...")
    anim.save(str(output_path), writer='pillow', fps=fps)
    plt.close()

    print(f"Done! Animation saved to {output_path}")
    return output_path


def main():
    # Animate morning rush hour on selected lines
    create_animation(
        lines_to_animate=['BAK', 'VIC', 'CEN', 'NTN', 'PIC'],
        start_time_str='08:00:00',
        duration_minutes=30,  # 30 minutes of real time
        fps=10,
        speed_multiplier=60  # 1 minute real = 1 second animation
    )


if __name__ == '__main__':
    main()
