"""
Process TransXChange XML files to feather format.

This script parses TfL TransXChange timetable data and extracts:
- StopPoints (stations)
- RouteLinks (connections between stations)
- JourneyPatterns
- JourneyPatternTimingLinks
- VehicleJourneys
- And other related data

Usage:
    python process_transxchange.py              # Process all lines
    python process_transxchange.py DLR TRM      # Process specific lines
"""
import xml.etree.ElementTree as ET
import pandas as pd
import pyarrow.feather as feather
from pathlib import Path
from collections import defaultdict
import sys

# Paths
BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / 'data' / 'raw_data'
OUTPUT_DIR = BASE_DIR / 'data' / 'processed_data'
OUTPUT_DIR.mkdir(exist_ok=True)

# TransXChange namespace
NS = {'txc': 'http://www.transxchange.org.uk/'}

# Mapping of file prefixes to line codes
LINE_MAPPINGS = {
    'tfl_1-BAK': 'BAK',
    'tfl_1-CEN': 'CEN',
    'tfl_1-CIR': 'CIR',
    'tfl_1-DIS': 'DIS',
    'tfl_1-HAM': 'HAM',
    'tfl_1-JUB': 'JUB',
    'tfl_1-MET': 'MET',
    'tfl_1-NTN': 'NTN',
    'tfl_1-PIC': 'PIC',
    'tfl_1-VIC': 'VIC',
    'tfl_1-WAC': 'WAC',
    'tfl_25-DLR': 'DLR',
    'tfl_63-TR': 'TRM',
    'tfl_71-CAB': 'CAB',  # Emirates Cable Car
}


def get_text(element, path, ns=NS):
    """Safely get text from an XML element."""
    el = element.find(path, ns)
    return el.text if el is not None else None


def parse_stop_points(root):
    """Parse StopPoints from TransXChange XML."""
    stops = []
    for sp in root.findall('.//txc:StopPoint', NS):
        stop = {
            'AtcoCode': get_text(sp, 'txc:AtcoCode'),
            'Descriptor_CommonName': get_text(sp, 'txc:Descriptor/txc:CommonName'),
            'Place_NptgLocalityRef': get_text(sp, 'txc:Place/txc:NptgLocalityRef'),
            'Place_Location_Easting': get_text(sp, 'txc:Place/txc:Location/txc:Easting'),
            'Place_Location_Northing': get_text(sp, 'txc:Place/txc:Location/txc:Northing'),
        }
        if stop['AtcoCode']:
            stops.append(stop)
    return stops


def parse_route_links(root):
    """Parse RouteLinks from TransXChange XML."""
    links = []
    for rl in root.findall('.//txc:RouteLink', NS):
        link = {
            'RouteLink_id': rl.get('id'),
            'From_StopPointRef': get_text(rl, 'txc:From/txc:StopPointRef'),
            'To_StopPointRef': get_text(rl, 'txc:To/txc:StopPointRef'),
            'Distance': get_text(rl, 'txc:Distance'),
        }
        if link['From_StopPointRef'] and link['To_StopPointRef']:
            links.append(link)
    return links


def parse_routes(root):
    """Parse Routes from TransXChange XML."""
    routes = []
    for route in root.findall('.//txc:Route', NS):
        r = {
            'Route_id': route.get('id'),
            'PrivateCode': get_text(route, 'txc:PrivateCode'),
            'Description': get_text(route, 'txc:Description'),
        }
        routes.append(r)
    return routes


def parse_journey_pattern_sections(root):
    """Parse JourneyPatternSections and their timing links."""
    timing_links = []
    for jps in root.findall('.//txc:JourneyPatternSection', NS):
        section_id = jps.get('id')
        seq = 0
        for jptl in jps.findall('txc:JourneyPatternTimingLink', NS):
            seq += 1
            link = {
                'JourneyPatternSections': section_id,
                'JourneyPatternTimingLink_id': jptl.get('id'),
                'From_StopPointRef': get_text(jptl, 'txc:From/txc:StopPointRef'),
                'From_SequenceNumber': get_text(jptl, 'txc:From/txc:SequenceNumber') or seq,
                'To_StopPointRef': get_text(jptl, 'txc:To/txc:StopPointRef'),
                'To_SequenceNumber': get_text(jptl, 'txc:To/txc:SequenceNumber') or seq + 1,
                'RouteLinkRef': get_text(jptl, 'txc:RouteLinkRef'),
                'RunTime': get_text(jptl, 'txc:RunTime'),
                'WaitTime': get_text(jptl, 'txc:From/txc:WaitTime'),
            }
            timing_links.append(link)
    return timing_links


def parse_journey_patterns(root):
    """Parse JourneyPatterns from TransXChange XML."""
    patterns = []
    for jp in root.findall('.//txc:JourneyPattern', NS):
        pattern = {
            'JourneyPattern': jp.get('id'),
            'DestinationDisplay': get_text(jp, 'txc:DestinationDisplay'),
            'Direction': get_text(jp, 'txc:Direction'),
            'RouteRef': get_text(jp, 'txc:RouteRef'),
            'JourneyPatternSectionRefs': get_text(jp, 'txc:JourneyPatternSectionRefs'),
        }
        patterns.append(pattern)
    return patterns


def parse_vehicle_journeys(root):
    """Parse VehicleJourneys from TransXChange XML."""
    journeys = []
    for vj in root.findall('.//txc:VehicleJourney', NS):
        journey = {
            'VehicleJourney_id': vj.get('id'),
            'VehicleJourneyCode': get_text(vj, 'txc:VehicleJourneyCode'),
            'ServiceRef': get_text(vj, 'txc:ServiceRef'),
            'LineRef': get_text(vj, 'txc:LineRef'),
            'JourneyPatternRef': get_text(vj, 'txc:JourneyPatternRef'),
            'DepartureTime': get_text(vj, 'txc:DepartureTime'),
        }
        journeys.append(journey)
    return journeys


def parse_services(root):
    """Parse Services from TransXChange XML."""
    services = []
    for svc in root.findall('.//txc:Service', NS):
        service = {
            'ServiceCode': get_text(svc, 'txc:ServiceCode'),
            'PrivateCode': get_text(svc, 'txc:PrivateCode'),
            'Mode': get_text(svc, 'txc:Mode'),
            'Description': get_text(svc, 'txc:Lines/txc:Line/txc:LineName'),
        }
        services.append(service)
    return services


def parse_nptg_localities(root):
    """Parse NptgLocalities from TransXChange XML."""
    localities = []
    for loc in root.findall('.//txc:AnnotatedNptgLocalityRef', NS):
        locality = {
            'NptgLocalityRef': get_text(loc, 'txc:NptgLocalityRef'),
            'LocalityName': get_text(loc, 'txc:LocalityName'),
        }
        if locality['NptgLocalityRef']:
            localities.append(locality)
    return localities


def process_line(line_code, xml_files):
    """Process all XML files for a single line."""
    print(f"\nProcessing {line_code} ({len(xml_files)} files)...")

    all_data = {
        'stops': [],
        'route_links': [],
        'routes': [],
        'timing_links': [],
        'patterns': [],
        'journeys': [],
        'services': [],
        'localities': [],
    }

    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            all_data['stops'].extend(parse_stop_points(root))
            all_data['route_links'].extend(parse_route_links(root))
            all_data['routes'].extend(parse_routes(root))
            all_data['timing_links'].extend(parse_journey_pattern_sections(root))
            all_data['patterns'].extend(parse_journey_patterns(root))
            all_data['journeys'].extend(parse_vehicle_journeys(root))
            all_data['services'].extend(parse_services(root))
            all_data['localities'].extend(parse_nptg_localities(root))

        except ET.ParseError as e:
            print(f"  Warning: Could not parse {xml_file.name}: {e}")

    # Convert to DataFrames and deduplicate
    results = {}

    if all_data['stops']:
        df = pd.DataFrame(all_data['stops']).drop_duplicates(subset=['AtcoCode'])
        results['StopPoints'] = df
        print(f"  StopPoints: {len(df)}")

    if all_data['route_links']:
        df = pd.DataFrame(all_data['route_links']).drop_duplicates()
        results['RouteLinks'] = df
        print(f"  RouteLinks: {len(df)}")

    if all_data['routes']:
        df = pd.DataFrame(all_data['routes']).drop_duplicates(subset=['Route_id'])
        results['Routes'] = df
        print(f"  Routes: {len(df)}")

    if all_data['timing_links']:
        df = pd.DataFrame(all_data['timing_links']).drop_duplicates()
        results['JourneyPatternTimingLinks'] = df
        print(f"  JourneyPatternTimingLinks: {len(df)}")

    if all_data['patterns']:
        df = pd.DataFrame(all_data['patterns']).drop_duplicates(subset=['JourneyPattern'])
        results['JourneyPatterns'] = df
        print(f"  JourneyPatterns: {len(df)}")

    if all_data['journeys']:
        df = pd.DataFrame(all_data['journeys']).drop_duplicates()
        results['VehicleJourneys'] = df
        print(f"  VehicleJourneys: {len(df)}")

    if all_data['services']:
        df = pd.DataFrame(all_data['services']).drop_duplicates()
        results['Services'] = df
        print(f"  Services: {len(df)}")

    if all_data['localities']:
        df = pd.DataFrame(all_data['localities']).drop_duplicates(subset=['NptgLocalityRef'])
        results['NptgLocalities'] = df
        print(f"  NptgLocalities: {len(df)}")

    return results


def save_line_data(line_code, data):
    """Save processed data to feather files."""
    for name, df in data.items():
        output_path = OUTPUT_DIR / f'{line_code}-{name}.feather'
        feather.write_feather(df, output_path)
        print(f"  Saved: {output_path.name}")


def update_all_stoppoints():
    """Merge all StopPoints into a single file."""
    all_stops = []
    for feather_file in OUTPUT_DIR.glob('*-StopPoints.feather'):
        df = feather.read_feather(feather_file)
        all_stops.append(df)

    if all_stops:
        combined = pd.concat(all_stops, ignore_index=True)
        combined = combined.drop_duplicates(subset=['AtcoCode'])
        output_path = OUTPUT_DIR / 'ALL-StopPoints.feather'
        feather.write_feather(combined, output_path)
        print(f"\nUpdated ALL-StopPoints.feather: {len(combined)} stations")


def main():
    # Determine which lines to process
    if len(sys.argv) > 1:
        lines_to_process = [arg.upper() for arg in sys.argv[1:]]
    else:
        lines_to_process = None  # Process all

    # Group XML files by line
    xml_by_line = defaultdict(list)
    for xml_file in RAW_DIR.glob('*.xml'):
        for prefix, line_code in LINE_MAPPINGS.items():
            if xml_file.name.startswith(prefix):
                xml_by_line[line_code].append(xml_file)
                break

    # Process each line
    for line_code, xml_files in sorted(xml_by_line.items()):
        if lines_to_process and line_code not in lines_to_process:
            continue

        data = process_line(line_code, xml_files)
        if data:
            save_line_data(line_code, data)

    # Update combined StopPoints
    update_all_stoppoints()

    print("\nDone!")


if __name__ == '__main__':
    main()
