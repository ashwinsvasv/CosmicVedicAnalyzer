from flask import Flask, request, render_template, redirect, url_for,abort,send_from_directory
from werkzeug.utils import secure_filename
import pandas as pd
try:
    from .astrology_analyzer import AstrologyAnalyzer
except:
    from astrology_analyzer import AstrologyAnalyzer
import io
import re
import os
from datetime import datetime
import numpy as np

app = Flask(__name__)

def extract_datetime_components(date_str):
    """Try to parse various date formats into components"""
    if not isinstance(date_str, str):
        return None
    
    # Clean the date string
    date_str = date_str.strip()
    
    try:
        # Try ISO format first
        dt = datetime.fromisoformat(date_str)
        return {
            'YEAR': dt.year,
            'MONTH': dt.month,
            'DAY': dt.day,
            'HOUR': dt.hour,
            'MINUTE': dt.minute
        }
    except (ValueError, AttributeError):
        pass
    
    # Try other common formats
    formats = [
        ('%Y-%m-%d %H:%M:%S', 1),
        ('%Y-%m-%d %H:%M', 1),
        ('%d-%m-%Y %H:%M:%S', 1),
        ('%d-%m-%Y %H:%M', 1),
        ('%m/%d/%Y %H:%M', 1),
        ('%d/%m/%Y %H:%M', 1),
        ('%d-%b-%Y %H:%M', 1),
        ('%d-%b-%Y %H:%M:%S', 1),
        ('%Y-%m-%d', 0),  # Date only
        ('%m/%d/%Y', 0),  # Date only
        ('%d/%m/%Y', 0),  # Date only
        ('%d-%m-%Y', 0),  # Date only
    ]
    
    for fmt, has_time in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return {
                'YEAR': dt.year,
                'MONTH': dt.month,
                'DAY': dt.day,
                'HOUR': dt.hour if has_time else 12,
                'MINUTE': dt.minute if has_time else 0
            }
        except ValueError:
            continue
    
    # Custom format for DD-MM-YYYY HH:MM
    try:
        date_pattern = r'(\d{2})-(\d{2})-(\d{4})\s+(\d{1,2}):(\d{2})'
        match = re.match(date_pattern, date_str)
        if match:
            day, month, year, hour, minute = match.groups()
            return {
                'YEAR': int(year),
                'MONTH': int(month),
                'DAY': int(day),
                'HOUR': int(hour),
                'MINUTE': int(minute)
            }
    except:
        pass
    
    # Try earthquake dataset specific format (DD-MM-YYYY HH:MM)
    try:
        date_pattern = r'(\d{1,2})-(\d{1,2})-(\d{4})\s+(\d{1,2}):(\d{1,2})'
        match = re.search(date_pattern, date_str)
        if match:
            day, month, year, hour, minute = match.groups()
            return {
                'YEAR': int(year),
                'MONTH': int(month),
                'DAY': int(day),
                'HOUR': int(hour),
                'MINUTE': int(minute)
            }
    except:
        pass
    
    return None

def extract_coordinates(location_str):
    """Extract lat/long from various location formats"""
    if not isinstance(location_str, str):
        return None, None
    
    # Try decimal degrees format
    decimal_pattern = r'(-?\d+\.\d+)[, ]+(-?\d+\.\d+)'
    match = re.search(decimal_pattern, location_str)
    if match:
        return float(match.group(1)), float(match.group(2))
    
    # Try DMS format
    dms_pattern = r'(\d+)°(\d+)\'(\d+\.?\d*)"([NS]),?\s*(\d+)°(\d+)\'(\d+\.?\d*)"([EW])'
    match = re.search(dms_pattern, location_str)
    if match:
        lat_deg, lat_min, lat_sec, lat_dir = match.group(1, 2, 3, 4)
        lon_deg, lon_min, lon_sec, lon_dir = match.group(5, 6, 7, 8)
        
        lat = float(lat_deg) + float(lat_min)/60 + float(lat_sec)/3600
        if lat_dir.upper() == 'S':
            lat = -lat
            
        lon = float(lon_deg) + float(lon_min)/60 + float(lon_sec)/3600
        if lon_dir.upper() == 'W':
            lon = -lon
            
        return lat, lon
    
    return None, None

def identify_column_types(df):
    """Identify different types of columns in the dataframe"""
    column_types = {
        'date': [],
        'time': [],
        'datetime': [],
        'magnitude': [],
        'latitude': [],
        'longitude': [],
        'location': [],
        'depth': []
    }
    
    # Check each column name for keywords
    for col in df.columns:
        col_lower = str(col).lower()
        
        # Check for date/time columns
        if any(kw in col_lower for kw in ['date_time', 'datetime']):
            column_types['datetime'].append(col)
        elif 'date' in col_lower:
            column_types['date'].append(col)
        elif 'time' in col_lower:
            column_types['time'].append(col)
            
        # Check for magnitude columns
        if any(kw in col_lower for kw in ['magnitude', 'mag', ' m ']):
            column_types['magnitude'].append(col)
            
        # Check for coordinate columns
        if any(kw == col_lower for kw in ['latitude', 'lat']):
            column_types['latitude'].append(col)
        if any(kw == col_lower for kw in ['longitude', 'long', 'lon']):
            column_types['longitude'].append(col)
            
        # Check for location columns
        if any(kw in col_lower for kw in ['location', 'place']):
            column_types['location'].append(col)
            
        # Check for depth columns
        if 'depth' in col_lower:
            column_types['depth'].append(col)
    
    # If we don't have explicit columns, try to identify by content
    if not column_types['magnitude']:
        for col in df.columns:
            # Try to identify magnitude columns (typically small numbers < 10)
            if df[col].dtype in [np.float64, np.int64]:
                values = pd.to_numeric(df[col], errors='coerce')
                if values.min() >= 0 and values.max() <= 10:
                    column_types['magnitude'].append(col)
                    break
    
    return column_types

def extract_magnitude(df):
    """Try to extract magnitude from various column formats"""
    # First check columns that might contain magnitude information
    magnitude_cols = [col for col in df.columns if any(kw in str(col).lower() for kw in ['magnitude', 'mag', ' m '])]
    
    if magnitude_cols:
        # Use the first detected magnitude column
        mag_col = magnitude_cols[0]
        return pd.to_numeric(df[mag_col], errors='coerce')
    
    # If no explicit magnitude column, look for a column with values between 0 and 10
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            try:
                values = pd.to_numeric(df[col], errors='coerce')
                if values.min() >= 0 and values.max() <= 10 and values.mean() > 2:
                    return values
            except:
                continue
    
    # If all else fails, try to extract from the first column if it's a string
    # (sometimes descriptions contain magnitude like "M 6.5")
    if len(df.columns) > 0 and df[df.columns[0]].dtype == object:
        try:
            patterns = [r'M\s*(\d+\.\d+)', r'magnitude\s*(\d+\.\d+)', r'mag\s*(\d+\.\d+)']
            for pattern in patterns:
                mag_series = df[df.columns[0]].str.extract(pattern, expand=False)
                if not mag_series.isna().all():
                    return pd.to_numeric(mag_series, errors='coerce')
        except:
            pass
    
    # Default to 0 if no magnitude found
    return pd.Series([0.0] * len(df))

def normalize_csv(df):
    """Normalize different CSV formats to our required structure"""
    normalized = pd.DataFrame()
    
    # Identify column types
    column_types = identify_column_types(df)
    
    # Handle datetime columns
    datetime_cols = column_types['datetime']
    date_cols = column_types['date']
    time_cols = column_types['time']
    
    # Process datetime information
    if datetime_cols:
        # If we have a datetime column, try to parse it
        datetime_col = datetime_cols[0]
        df[datetime_col] = df[datetime_col].astype(str)
        
        datetime_data = df[datetime_col].apply(extract_datetime_components)
        datetime_df = pd.DataFrame(datetime_data.tolist())
        
        for col in ['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE']:
            if col in datetime_df.columns:
                normalized[col] = datetime_df[col]
            else:
                normalized[col] = 1 if col in ['MONTH', 'DAY'] else (12 if col == 'HOUR' else 0)
    
    elif date_cols:
        # If we have separate date and time columns
        date_col = date_cols[0]
        df[date_col] = df[date_col].astype(str)
        
        # Parse date components
        if time_cols:
            # Combine date and time for parsing
            time_col = time_cols[0]
            df['temp_datetime'] = df[date_col] + ' ' + df[time_col].astype(str)
            datetime_data = df['temp_datetime'].apply(extract_datetime_components)
        else:
            datetime_data = df[date_col].apply(extract_datetime_components)
            
        datetime_df = pd.DataFrame(datetime_data.tolist())
        
        for col in ['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE']:
            if col in datetime_df.columns:
                normalized[col] = datetime_df[col]
            else:
                normalized[col] = 1 if col in ['MONTH', 'DAY'] else (12 if col == 'HOUR' else 0)
    
    else:
        # Try to find individual components
        for col in ['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE']:
            if col in df.columns:
                normalized[col] = pd.to_numeric(df[col], errors='coerce').fillna(1 if col in ['MONTH', 'DAY'] else (12 if col == 'HOUR' else 0))
            else:
                normalized[col] = 1 if col in ['MONTH', 'DAY'] else (12 if col == 'HOUR' else 0)
    
    # Handle location data
    lat_cols = column_types['latitude']
    lon_cols = column_types['longitude']
    loc_cols = column_types['location']
    
    if lat_cols and lon_cols:
        # If we have explicit latitude and longitude columns
        normalized['LATITUDE'] = pd.to_numeric(df[lat_cols[0]], errors='coerce')
        normalized['LONGITUDE'] = pd.to_numeric(df[lon_cols[0]], errors='coerce')
    elif loc_cols:
        # Try to extract coordinates from location string
        loc_col = loc_cols[0]
        df[loc_col] = df[loc_col].astype(str)
        
        # Apply function to extract coordinates from location string
        coords = df[loc_col].apply(lambda x: pd.Series(extract_coordinates(x)))
        normalized['LATITUDE'] = coords[0]
        normalized['LONGITUDE'] = coords[1]
    else:
        # Check if we have columns that might be latitude/longitude
        for col in df.columns:
            col_str = str(col).lower()
            # Look for explicit latitude/longitude columns
            if 'latitude' in col_str or 'lat' == col_str:
                normalized['LATITUDE'] = pd.to_numeric(df[col], errors='coerce')
            elif 'longitude' in col_str or 'lon' == col_str or 'long' == col_str:
                normalized['LONGITUDE'] = pd.to_numeric(df[col], errors='coerce')
    
    # If we still don't have coordinates, default to 0,0
    if 'LATITUDE' not in normalized.columns:
        normalized['LATITUDE'] = 0.0
    if 'LONGITUDE' not in normalized.columns:
        normalized['LONGITUDE'] = 0.0
    
    # Extract magnitude
    normalized['MAGNITUDE'] = extract_magnitude(df)
    
    # Extract depth if available
    depth_cols = column_types['depth']
    if depth_cols:
        normalized['DEPTH'] = pd.to_numeric(df[depth_cols[0]], errors='coerce')
    else:
        # Try to find depth in any numeric column
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                col_str = str(col).lower()
                if 'depth' in col_str:
                    normalized['DEPTH'] = pd.to_numeric(df[col], errors='coerce')
                    break
        
        # Default depth if not found
        if 'DEPTH' not in normalized.columns:
            normalized['DEPTH'] = 0.0
    
    # Add location description if available
    if loc_cols:
        normalized['LOCATION'] = df[loc_cols[0]]
    
    # Drop rows with null values in critical columns
    normalized = normalized.dropna(subset=['YEAR', 'MONTH', 'DAY'])
    
    # Fill missing coordinates with defaults
    normalized['LATITUDE'].fillna(0.0, inplace=True)
    normalized['LONGITUDE'].fillna(0.0, inplace=True)
    
    return normalized

def detect_format(df):
    """Detect the format of the uploaded CSV"""
    format_info = {
        'type': 'unknown',
        'columns': df.columns.tolist(),
        'example_row': df.iloc[0].to_dict() if len(df) > 0 else {}
    }
    
    # Check for earthquake data format
    earthquake_indicators = ['magnitude', 'mag', 'depth', 'tsunami', 'mmi', 'cdi', 'sig']
    earthquake_score = sum(1 for col in df.columns if any(ind in str(col).lower() for ind in earthquake_indicators))
    
    if earthquake_score >= 2:
        format_info['type'] = 'earthquake'
    
    return format_info

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file uploaded')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No file selected')
        
        if file:
            try:
                # Read file directly without saving
                file_content = file.read().decode('utf-8')
                file_io = io.StringIO(file_content)
                
                # Try multiple CSV parsers with different delimiters
                tried_parsers = []
                
                for delimiter in [',', ';', '\t', '|']:
                    try:
                        df = pd.read_csv(file_io, sep=delimiter)
                        if len(df.columns) > 1:  # Successfully parsed multiple columns
                            break
                        tried_parsers.append(f"Delimiter '{delimiter}': {len(df.columns)} columns")
                        file_io.seek(0)  # Reset file pointer for next attempt
                    except Exception as e:
                        tried_parsers.append(f"Delimiter '{delimiter}': {str(e)}")
                        file_io.seek(0)  # Reset file pointer for next attempt
                
                # If pandas couldn't parse it, try manual parsing
                if len(df.columns) <= 1 and ' ' in file_content:
                    # Try to split by whitespace and create columns
                    lines = file_content.strip().split('\n')
                    if len(lines) > 1:
                        # Use first line as header if it doesn't contain numbers
                        if not any(c.isdigit() for c in lines[0]):
                            header = lines[0].split()
                            data = [line.split() for line in lines[1:]]
                        else:
                            # Generate column names
                            max_cols = max(len(line.split()) for line in lines)
                            header = [f'Column{i+1}' for i in range(max_cols)]
                            data = [line.split() for line in lines]
                        
                        # Create DataFrame
                        df = pd.DataFrame(data, columns=header[:len(data[0])])
                
                # Detect format
                format_info = detect_format(df)
                
                # Normalize the dataframe
                normalized_df = normalize_csv(df)
                
                if len(normalized_df) == 0:
                    return render_template('index.html', error='No valid data found in the file')
                
                # Process with analyzer
                analyzer = AstrologyAnalyzer(df=normalized_df)
                results = analyzer.analyze_events()
                
                return render_template('index.html', results=results, format_info=format_info)
                
            except Exception as e:
                return render_template('index.html', error=f'Error processing file: {str(e)}')
    
    return render_template('index.html')

DATASETS_DIR = os.path.join(app.static_folder, './')

# Route to serve sample dataset files
@app.route('/download/<filename>')
def download_file(filename):
    # Define allowed files to prevent unauthorized access
    allowed_files = [
        'earthquake_1995-2023.csv',
        'tsunami_dataset.csv'
    ]
    
    # Check if the requested file is allowed
    if filename not in allowed_files:
        abort(404)  # Return 404 if file is not in the allowed list
    
    # Check if the file exists
    file_path = os.path.join(DATASETS_DIR, filename)
    if not os.path.isfile(file_path):
        abort(404)  # Return 404 if file doesn't exist
    
    # Serve the file
    return send_from_directory(DATASETS_DIR, filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)