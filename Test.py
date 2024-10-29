import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
import matplotlib.pyplot as plt
from io import BytesIO
import os
from datetime import datetime
from math import ceil
from geopy.distance import geodesic


def load_traffic_data(csv_path):
    """
    Loads traffic accident data from a CSV file with the specific CDMX structure
    """
    try:
        # Read the CSV file with specific column names
        df = pd.read_csv(csv_path, encoding='utf-8', low_memory=False)

        # Rename columns to standardize
        column_mapping = {
            'latitud': 'latitude',
            'longitud': 'longitude',
            'fecha_evento': 'date',
            'hora_evento': 'time',
            'tipo_evento': 'event_type',
            'alcaldia': 'district',
            'colonia': 'neighborhood',
            'personas_fallecidas': 'fatalities',
            'personas_lesionadas': 'injuries'
        }
        df = df.rename(columns=column_mapping)

        # Convert coordinates to numeric
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

        # Convert date and time
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')

        # Convert numeric fields
        df['fatalities'] = pd.to_numeric(df['fatalities'], errors='coerce').fillna(0)
        df['injuries'] = pd.to_numeric(df['injuries'], errors='coerce').fillna(0)

        # Drop rows with invalid coordinates
        df = df.dropna(subset=['latitude', 'longitude'])

        print(f"Successfully loaded {len(df)} valid records from CSV")
        return df

    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None


def create_time_distribution(data, title):
    """
    Creates a time distribution plot
    """
    plt.figure(figsize=(8, 4))
    data['hour'] = data['datetime'].dt.hour
    hourly_dist = data['hour'].value_counts().sort_index()
    plt.bar(hourly_dist.index, hourly_dist.values)
    plt.title(title)
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Accidents')
    plt.grid(True, alpha=0.3)

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    return buf


def calculate_grid_size(bounds, target_cell_size_km=0.5):
    """
    Calculates appropriate grid dimensions based on area and target cell size
    bounds: tuple of (minx, miny, maxx, maxy) in degrees
    target_cell_size_km: target size of each cell in kilometers
    Returns: tuple of (n_rows, n_cols)
    """
    minx, miny, maxx, maxy = bounds

    # Ensure coordinates are in (latitude, longitude) order
    width_km = geodesic((miny, minx), (miny, maxx)).kilometers
    height_km = geodesic((miny, minx), (maxy, minx)).kilometers

    # Calculate number of cells needed
    n_cols = ceil(width_km / target_cell_size_km)
    n_rows = ceil(height_km / target_cell_size_km)

    # Print information about the grid
    print(f"Area dimensions: {width_km:.1f} km x {height_km:.1f} km")
    print(f"Target cell size: {target_cell_size_km} km x {target_cell_size_km} km")
    print(f"Grid size: {n_rows} rows x {n_cols} columns ({n_rows * n_cols} cells)")
    print(f"Actual cell size: {width_km / n_cols:.2f} km x {height_km / n_rows:.2f} km")

    return n_rows, n_cols


def divide_area_into_grid(bounds, target_cell_size_km=0.5):
    """
    Divides the study area into a grid based on target cell size in kilometers
    bounds: tuple of (minx, miny, maxx, maxy)
    Returns: GeoDataFrame with grid cells and their information
    """
    n_rows, n_cols = calculate_grid_size(bounds, target_cell_size_km)
    minx, miny, maxx, maxy = bounds

    width = (maxx - minx) / n_cols
    height = (maxy - miny) / n_rows

    grid_cells = []
    cell_info = []

    for i in range(n_rows):
        for j in range(n_cols):
            cell_minx = minx + j * width
            cell_miny = miny + i * height
            cell = box(cell_minx, cell_miny, cell_minx + width, cell_miny + height)

            # Calculate center point of cell
            center_x = cell_minx + width / 2
            center_y = cell_miny + height / 2

            grid_cells.append(cell)
            cell_info.append({
                'row': i + 1,
                'col': j + 1,
                'bounds': (cell_minx, cell_miny, cell_minx + width, cell_miny + height),
                'center_lat': center_y,
                'center_lon': center_x,
                'cell_width_km': width * 104.647,  # approximate conversion to km
                'cell_height_km': height * 110.574
            })

    grid_df = gpd.GeoDataFrame(cell_info, geometry=grid_cells)
    return grid_df

def create_heatmap(data, title):
    """
    Creates a heatmap visualization
    """
    plt.figure(figsize=(8, 6))
    plt.hist2d(data['longitude'], data['latitude'], bins=20, cmap='YlOrRd')
    plt.colorbar(label='Number of accidents')
    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    return buf


def generate_area_report(data, grid_cell, cell_info, output_dir):
    """
    Generates a PDF report for a specific grid cell
    """
    cell_id = f"r{cell_info['row']}c{cell_info['col']}"
    pdf_path = os.path.join(output_dir, f'area_report_{cell_id}.pdf')
    c = canvas.Canvas(pdf_path, pagesize=letter)

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 750, f"Traffic Analysis Report - Area {cell_id}")

    # Area bounds
    c.setFont("Helvetica", 10)
    bounds = cell_info['bounds']
    c.drawString(50, 730, f"Area Bounds: {bounds[0]:.4f}W, {bounds[1]:.4f}S, {bounds[2]:.4f}E, {bounds[3]:.4f}N")

    if not data.empty:
        y_position = 700
        c.setFont("Helvetica-Bold", 12)

        # General statistics
        total_accidents = len(data)
        total_fatalities = int(data['fatalities'].sum())
        total_injuries = int(data['injuries'].sum())

        c.drawString(50, y_position, "General Statistics")
        y_position -= 20
        c.setFont("Helvetica", 11)
        c.drawString(70, y_position, f"Total accidents: {total_accidents}")
        y_position -= 15
        c.drawString(70, y_position, f"Total fatalities: {total_fatalities}")
        y_position -= 15
        c.drawString(70, y_position, f"Total injuries: {total_injuries}")
        y_position -= 25

        # District distribution
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y_position, "Distribution by District")
        y_position -= 20
        c.setFont("Helvetica", 11)
        district_counts = data['district'].value_counts()
        for district, count in district_counts.head(5).items():
            c.drawString(70, y_position, f"{district}: {count} accidents")
            y_position -= 15
        y_position -= 10

        # Event type distribution
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y_position, "Distribution by Event Type")
        y_position -= 20
        c.setFont("Helvetica", 11)
        type_counts = data['event_type'].value_counts()
        for event_type, count in type_counts.head(5).items():
            c.drawString(70, y_position, f"{event_type}: {count} accidents")
            y_position -= 15
        y_position -= 10

        # Time distribution plot
        if len(data) > 5:
            try:
                time_dist_buf = create_time_distribution(data, f"Hourly Distribution - Area {cell_id}")
                c.drawImage(time_dist_buf, 50, y_position - 200, width=400, height=200)
                y_position -= 210
            except Exception as e:
                print(f"Error creating time distribution plot for area {cell_id}: {e}")

        # Heatmap
        if len(data) > 5:
            try:
                heatmap_buf = create_heatmap(data, f"Accident Heatmap - Area {cell_id}")
                c.drawImage(heatmap_buf, 50, y_position - 300, width=400, height=300)
            except Exception as e:
                print(f"Error creating heatmap for area {cell_id}: {e}")

    c.save()


def main():
    # Get input from user
    csv_path = "C:/Users/Jovany/PycharmProjects/pythonProject/nuevo_acumulado_hechos_de_transito_2023_12.csv"

    # Load data
    print("Loading traffic data...")
    data = load_traffic_data(csv_path)
    if data is None:
        print("Failed to load data. Exiting.")
        return

    # Calculate bounds from data
    print("\nCalculating analysis area...")
    margin = 0.01  # Add a small margin around the data
    bounds = (
        data['longitude'].min() - margin,
        data['latitude'].min() - margin,
        data['longitude'].max() + margin,
        data['latitude'].max() + margin
    )

    # Get cell size from user
    print("\nRecommended cell sizes:")
    print("0.5 km - Very detailed analysis (neighborhood level)")
    print("1.0 km - Detailed analysis (small district level)")
    print("2.0 km - Medium detail (large district level)")
    target_cell_size = float(input("\nEnter target cell size in kilometers (default 0.5): ") or 0.5)

    # Create grid
    print("\nCreating analysis grid...")
    grid = divide_area_into_grid(bounds, target_cell_size)

    # Convert data to GeoDataFrame
    try:
        gdf = gpd.GeoDataFrame(
            data,
            geometry=gpd.points_from_xy(data['longitude'], data['latitude'])
        )
    except Exception as e:
        print(f"Error creating GeoDataFrame: {e}")
        return

    # Create output directory
    output_dir = "traffic_reports"
    os.makedirs(output_dir, exist_ok=True)

    # Generate reports for each grid cell
    print("\nGenerating reports...")
    total_cells = len(grid)
    cells_with_data = 0

    for idx, row in grid.iterrows():
        cell_data = gdf[gdf.geometry.within(row.geometry)]
        if len(cell_data) > 0:
            cells_with_data += 1
            generate_area_report(cell_data, row.geometry, row, output_dir)
            print(f"Generated report for area r{row['row']}c{row['col']} ({len(cell_data)} accidents)")

    print(f"\nAnalysis complete:")
    print(f"Total grid cells: {total_cells}")
    print(f"Cells with accidents: {cells_with_data}")
    print(f"Coverage: {(cells_with_data / total_cells * 100):.1f}%")
    print(f"Reports generated in {output_dir}")


if __name__ == "__main__":
    main()
