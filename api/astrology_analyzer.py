import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ephem import Moon, Sun, Mercury,Venus,Mars,Jupiter,Saturn,Observer
import math
from collections import Counter, defaultdict
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import networkx as nx
from scipy.stats import chi2_contingency
from itertools import combinations

# Vedic astrological constants with Sanskrit names
SIGNS = [
    'Mesha', 'Vrishabha', 'Mithuna', 'Karka',
    'Simha', 'Kanya', 'Tula', 'Vrishchika',
    'Dhanus', 'Makara', 'Kumbha', 'Meena'
]

PLANETS = {
    'Ravi': Sun,
    'Chandra': Moon,
    'Budha': Mercury,
    'Shukra': Venus,
    'Kuja': Mars,
    'Guru': Jupiter,
    'Shani': Saturn,
}

PLANET_NAMES = list(PLANETS.keys()) + ['Rahu', 'Ketu']
AYANAMSA = 24.0  # Lahiri ayanamsa approximation
HOUSES = list(range(1, 13))  # 12 houses

# Vedic aspects and their orbs
VEDIC_ASPECTS = {
    'Conjunction': {'angle': 0, 'orb': 10},
    'Opposition': {'angle': 180, 'orb': 10},
    'Trine': {'angle': 120, 'orb': 10},
    'Square': {'angle': 90, 'orb': 10},
    'Sextile': {'angle': 60, 'orb': 6},
    'Quincunx': {'angle': 150, 'orb': 3},
    'Semi-Sextile': {'angle': 30, 'orb': 3}
}

# Planet special aspects (in degrees)
SPECIAL_ASPECTS = {
    'Guru': [120, 180, 240],  # Jupiter aspects 5th, 7th, 9th
    'Kuja': [90, 180, 240],   # Mars aspects 4th, 7th, 8th
    'Shani': [90, 180, 270],  # Saturn aspects 3rd, 7th, 10th
    'Rahu': [120, 180, 240],  # Rahu aspects 5th, 7th, 9th
    'Ketu': [120, 180, 240]   # Ketu aspects 5th, 7th, 9th
}

# Planetary relationships (friends, enemies, neutral)
PLANET_RELATIONSHIPS = {
    'Ravi': {'friends': ['Chandra', 'Kuja', 'Guru'], 'enemies': ['Shani', 'Rahu', 'Ketu'], 'neutral': ['Budha', 'Shukra']},
    'Chandra': {'friends': ['Ravi', 'Budha'], 'enemies': ['Rahu', 'Ketu'], 'neutral': ['Kuja', 'Guru', 'Shukra', 'Shani']},
    'Budha': {'friends': ['Ravi', 'Shukra'], 'enemies': ['Chandra'], 'neutral': ['Kuja', 'Guru', 'Shani', 'Rahu', 'Ketu']},
    'Shukra': {'friends': ['Budha', 'Shani'], 'enemies': ['Ravi', 'Chandra'], 'neutral': ['Kuja', 'Guru', 'Rahu', 'Ketu']},
    'Kuja': {'friends': ['Ravi', 'Chandra', 'Guru'], 'enemies': ['Budha'], 'neutral': ['Shukra', 'Shani', 'Rahu', 'Ketu']},
    'Guru': {'friends': ['Ravi', 'Chandra', 'Kuja'], 'enemies': ['Budha', 'Shukra'], 'neutral': ['Shani', 'Rahu', 'Ketu']},
    'Shani': {'friends': ['Budha', 'Shukra'], 'enemies': ['Ravi', 'Chandra', 'Kuja'], 'neutral': ['Guru', 'Rahu', 'Ketu']},
    'Rahu': {'friends': ['Shukra', 'Shani'], 'enemies': ['Ravi', 'Chandra'], 'neutral': ['Budha', 'Kuja', 'Guru', 'Ketu']},
    'Ketu': {'friends': ['Kuja', 'Shani'], 'enemies': ['Ravi', 'Chandra'], 'neutral': ['Budha', 'Shukra', 'Guru', 'Rahu']}
}

class AstrologyAnalyzer:
    def __init__(self, csv_file=None, df=None):
        """Initialize with either a CSV file or a pandas DataFrame"""
        if df is not None:
            self.df = df
        elif csv_file:
            self.df = pd.read_csv(csv_file)
        else:
            raise ValueError("Either csv_file or df must be provided")
        
        # Initialize columns for planet positions, signs, and houses
        for planet in PLANET_NAMES:
            self.df[f"{planet}_POS"] = None
            self.df[f"{planet}_SIGN"] = None
            self.df[f"{planet}_HOUSE"] = None
        
        self.df['ASCENDANT'] = None
        self.df['ASCENDANT_SIGN'] = None
        
        # Initialize aspect columns
        self.aspect_columns = []
        for p1, p2 in combinations(PLANET_NAMES, 2):
            col_name = f"{p1}_{p2}_ASPECT"
            self.df[col_name] = None
            self.aspect_columns.append(col_name)
        
        # Initialize planetary strength columns
        for planet in PLANET_NAMES:
            self.df[f"{planet}_STRENGTH"] = None
            
        self.preprocess_data()

    def preprocess_data(self):
        """Clean and preprocess the event dataset"""
        date_cols = ['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE']
        for col in date_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Fill missing values with defaults
        self.df['MONTH'] = self.df['MONTH'].fillna(1).astype(int)
        self.df['DAY'] = self.df['DAY'].fillna(1).astype(int)
        self.df['HOUR'] = self.df['HOUR'].fillna(12).astype(int)
        self.df['MINUTE'] = self.df['MINUTE'].fillna(0).astype(int)
        
        # Ensure latitude and longitude are available
        self.df = self.df.dropna(subset=['LATITUDE', 'LONGITUDE'])
        self.df['LATITUDE'] = self.df['LATITUDE'].astype(float)
        self.df['LONGITUDE'] = self.df['LONGITUDE'].astype(float)
        
        # Add event category column if it doesn't exist
        if 'EVENT_CATEGORY' not in self.df.columns:
            self.df['EVENT_CATEGORY'] = 'Unknown'

    def degrees_to_dms(self, decimal_degrees):
        """Convert decimal degrees to degrees, minutes, seconds format for ephem"""
        is_negative = decimal_degrees < 0
        decimal_degrees = abs(decimal_degrees)
        degrees = int(decimal_degrees)
        decimal_minutes = (decimal_degrees - degrees) * 60
        minutes = int(decimal_minutes)
        seconds = (decimal_minutes - minutes) * 60
        
        if is_negative:
            if degrees > 0:
                degrees = -degrees
            elif minutes > 0:
                minutes = -minutes
            else:
                seconds = -seconds
                
        return f"{degrees}:{minutes}:{seconds}"

    def create_observer(self, lat, lon, date):
        """Create ephem observer at the given location and time"""
        observer = Observer()
        observer.lat = self.degrees_to_dms(lat)
        observer.lon = self.degrees_to_dms(lon)
        observer.date = date
        return observer

    def calculate_planet_positions(self, observer):
        """Calculate sidereal positions of all planets for a given observer"""
        positions = {}
        
        # Calculate positions for standard planets
        for planet_name, planet_class in PLANETS.items():
            planet = planet_class()
            planet.compute(observer)
            lon_degrees = math.degrees(planet.ra) * 15  # Convert RA hours to degrees
            sidereal_lon = (lon_degrees - AYANAMSA) % 360
            positions[planet_name] = sidereal_lon
        
        # Calculate Rahu and Ketu
        moon = Moon()
        moon.compute(observer)
        rahu_lon = (math.degrees(moon.ra) * 15 + 90) % 360
        rahu_sidereal = (rahu_lon - AYANAMSA) % 360
        positions['Rahu'] = rahu_sidereal
        positions['Ketu'] = (rahu_sidereal + 180) % 360
        
        return positions

    def calculate_ascendant(self, observer):
        """Calculate ascendant for the given observer"""
        st_degrees = math.degrees(observer.sidereal_time()) * 15
        ascendant = (st_degrees - AYANAMSA) % 360
        return ascendant

    def get_sign(self, longitude):
        """Get Vedic zodiac sign from longitude"""
        if longitude is None or not isinstance(longitude, (int, float)):
            return "Unknown"
        sign_num = int(longitude / 30)
        return SIGNS[sign_num % 12]

    def get_house(self, longitude, asc_longitude):
        """Calculate house position based on Ascendant"""
        if longitude is None or asc_longitude is None:
            return None
        rel_pos = (longitude - asc_longitude) % 360
        house = int(rel_pos / 30) + 1
        return house

    def calculate_aspects(self, positions):
        """Calculate detailed Vedic aspects between planets"""
        aspects = {}
        
        # Initialize aspect dictionary with empty lists for each planet
        for planet in PLANET_NAMES:
            aspects[planet] = []
        
        # Calculate aspects between each pair of planets
        for p1, pos1 in positions.items():
            for p2, pos2 in positions.items():
                if p1 == p2:
                    continue
                    
                # Calculate angular separation
                sep = abs((pos1 - pos2) % 360)
                if sep > 180:
                    sep = 360 - sep
                
                # Check for standard aspects
                for aspect_name, aspect_data in VEDIC_ASPECTS.items():
                    if abs(sep - aspect_data['angle']) <= aspect_data['orb']:
                        aspects[p1].append({
                            'planet': p2, 
                            'aspect': aspect_name, 
                            'orb': round(abs(sep - aspect_data['angle']), 2)
                        })
                
                # Check for special aspects
                if p1 in SPECIAL_ASPECTS:
                    for special_angle in SPECIAL_ASPECTS[p1]:
                        if abs(sep - special_angle) <= 10:  # 10-degree orb for special aspects
                            house_diff = int(special_angle / 30)
                            aspects[p1].append({
                                'planet': p2,
                                'aspect': f"{house_diff}th house special aspect",
                                'orb': round(abs(sep - special_angle), 2)
                            })
        
        return aspects

    def calculate_planetary_strength(self, planet, position, house, ascendant_sign):
        """Calculate simplified Shadbala (planetary strength)"""
        strength = 0
        sign = self.get_sign(position)
        
        # Exaltation and debilitation
        exaltation_points = {
            'Ravi': 'Mesha',     # Sun exalted in Aries
            'Chandra': 'Vrishabha',  # Moon exalted in Taurus
            'Budha': 'Kanya',    # Mercury exalted in Virgo
            'Shukra': 'Meena',   # Venus exalted in Pisces
            'Kuja': 'Makara',    # Mars exalted in Capricorn
            'Guru': 'Karka',     # Jupiter exalted in Cancer
            'Shani': 'Tula',     # Saturn exalted in Libra
            'Rahu': 'Vrishabha', # Rahu strong in Taurus
            'Ketu': 'Vrishchika' # Ketu strong in Scorpio
        }
        
        debilitation_points = {
            'Ravi': 'Tula',      # Sun debilitated in Libra
            'Chandra': 'Vrishchika',  # Moon debilitated in Scorpio
            'Budha': 'Meena',    # Mercury debilitated in Pisces
            'Shukra': 'Kanya',   # Venus debilitated in Virgo
            'Kuja': 'Karka',     # Mars debilitated in Cancer
            'Guru': 'Makara',    # Jupiter debilitated in Capricorn
            'Shani': 'Mesha',    # Saturn debilitated in Aries
            'Rahu': 'Vrishchika', # Rahu weak in Scorpio
            'Ketu': 'Vrishabha'  # Ketu weak in Taurus
        }
        
        # Own sign rulerships
        own_signs = {
            'Ravi': ['Simha'],              # Sun rules Leo
            'Chandra': ['Karka'],           # Moon rules Cancer
            'Budha': ['Mithuna', 'Kanya'],  # Mercury rules Gemini and Virgo
            'Shukra': ['Vrishabha', 'Tula'], # Venus rules Taurus and Libra
            'Kuja': ['Mesha', 'Vrishchika'], # Mars rules Aries and Scorpio
            'Guru': ['Dhanus', 'Meena'],    # Jupiter rules Sagittarius and Pisces
            'Shani': ['Makara', 'Kumbha'],  # Saturn rules Capricorn and Aquarius
            'Rahu': [],                     # Rahu has no rulership
            'Ketu': []                      # Ketu has no rulership
        }
        
        # Add points for position
        if sign == exaltation_points.get(planet):
            strength += 5  # Exalted planet
        elif sign == debilitation_points.get(planet):
            strength -= 5  # Debilitated planet
        elif sign in own_signs.get(planet, []):
            strength += 3  # Planet in own sign
            
        # Add points for house position
        # Kendra houses (1, 4, 7, 10) are strong
        if house in [1, 4, 7, 10]:
            strength += 2
        # Trikona houses (1, 5, 9) are auspicious
        if house in [1, 5, 9]:
            strength += 2
        # Dusthana houses (6, 8, 12) are challenging
        if house in [6, 8, 12]:
            strength -= 2
            
        # Relationship with ascendant lord
        ascendant_lord = self.get_sign_ruler(ascendant_sign)
        if planet == ascendant_lord:
            strength += 3  # Planet is the ascendant lord
        elif ascendant_lord in PLANET_RELATIONSHIPS.get(planet, {}).get('friends', []):
            strength += 1  # Planet is friend of ascendant lord
        elif ascendant_lord in PLANET_RELATIONSHIPS.get(planet, {}).get('enemies', []):
            strength -= 1  # Planet is enemy of ascendant lord
            
        return strength

    def get_sign_ruler(self, sign):
        """Get the planetary ruler of a sign"""
        rulers = {
            'Mesha': 'Kuja',       # Aries ruled by Mars
            'Vrishabha': 'Shukra', # Taurus ruled by Venus
            'Mithuna': 'Budha',    # Gemini ruled by Mercury
            'Karka': 'Chandra',    # Cancer ruled by Moon
            'Simha': 'Ravi',       # Leo ruled by Sun
            'Kanya': 'Budha',      # Virgo ruled by Mercury
            'Tula': 'Shukra',      # Libra ruled by Venus
            'Vrishchika': 'Kuja',  # Scorpio ruled by Mars
            'Dhanus': 'Guru',      # Sagittarius ruled by Jupiter
            'Makara': 'Shani',     # Capricorn ruled by Saturn
            'Kumbha': 'Shani',     # Aquarius ruled by Saturn
            'Meena': 'Guru'        # Pisces ruled by Jupiter
        }
        return rulers.get(sign, None)

    def identify_yogas(self, positions, houses):
        """Identify key Vedic yogas (planetary combinations)"""
        yogas = []
        
        # Get planets by house
        planets_by_house = defaultdict(list)
        for planet, house in houses.items():
            if house:
                planets_by_house[house].append(planet)
        
        # Check for Raj Yoga (lord of 9th and 10th conjunct or aspecting)
        if planets_by_house.get(9) and planets_by_house.get(10):
            yogas.append({
                'name': 'Raj Yoga Influence',
                'description': 'Influence from 9th house (dharma) and 10th house (karma) connection',
                'strength': len(planets_by_house.get(9)) + len(planets_by_house.get(10))
            })
            
        # Check for Gajakesari Yoga (Jupiter and Moon in angular houses)
        if ('Guru' in houses and houses['Guru'] in [1, 4, 7, 10] and 
            'Chandra' in houses and houses['Chandra'] in [1, 4, 7, 10]):
            yogas.append({
                'name': 'Gajakesari Yoga',
                'description': 'Jupiter and Moon in angular positions from each other',
                'strength': 4
            })
            
        # Check for Budha-Aditya Yoga (Sun and Mercury together)
        for house, planet_list in planets_by_house.items():
            if 'Ravi' in planet_list and 'Budha' in planet_list:
                yogas.append({
                    'name': 'Budha-Aditya Yoga',
                    'description': 'Sun and Mercury together',
                    'strength': 3
                })
                break
                
        # Check for Chandra-Mangala Yoga (Moon and Mars together)
        for house, planet_list in planets_by_house.items():
            if 'Chandra' in planet_list and 'Kuja' in planet_list:
                yogas.append({
                    'name': 'Chandra-Mangala Yoga',
                    'description': 'Moon and Mars together',
                    'strength': 3
                })
                break
        
        return yogas

    def analyze_events(self, sample_size=None, detailed_patterns=True):
        """Calculate astrological positions and patterns with enhanced AI analysis"""
        if sample_size:
            events = self.df.sample(sample_size)
        else:
            events = self.df

        # Track house combinations, sign distributions, AI features, and aspects
        house_combinations = {house: Counter() for house in range(1, 13)}
        sign_distribution = {sign: Counter() for sign in SIGNS}
        aspect_patterns = Counter()
        yoga_patterns = Counter()
        feature_data = []
        processed_count = 0
        
        # Track categories if available
        category_data = {}
        if 'EVENT_CATEGORY' in self.df.columns:
            category_data = {
                'category_counts': Counter(),
                'category_planets': defaultdict(lambda: {p: Counter() for p in PLANET_NAMES}),
                'category_houses': defaultdict(lambda: {h: Counter() for h in range(1, 13)}),
                'category_signs': defaultdict(lambda: {s: Counter() for s in SIGNS}),
            }

        for idx, event in events.iterrows():
            try:
                # Create date string
                date_str = f"{int(event['YEAR'])}/{int(event['MONTH'])}/{int(event['DAY'])} {int(event['HOUR'])}:{int(event['MINUTE']):02d}"
                if event['YEAR'] <= 0:
                    date_str = f"{abs(int(event['YEAR'])-1)}/{int(event['MONTH'])}/{int(event['DAY'])} {int(event['HOUR'])}:{int(event['MINUTE']):02d} BC"
                
                # Create observer
                observer = self.create_observer(
                    event['LATITUDE'], 
                    event['LONGITUDE'], 
                    date_str
                )
                
                # Calculate ascendant
                ascendant = self.calculate_ascendant(observer)
                self.df.at[idx, 'ASCENDANT'] = ascendant
                ascendant_sign = self.get_sign(ascendant)
                self.df.at[idx, 'ASCENDANT_SIGN'] = ascendant_sign
                
                # Calculate planet positions
                positions = self.calculate_planet_positions(observer)
                
                # Calculate aspects between planets
                aspects = self.calculate_aspects(positions)
                
                # Track planets per house and sign
                house_planets = {house: [] for house in range(1, 13)}
                planet_houses = {}
                event_features = {
                    'ASCENDANT_SIGN': ascendant_sign,
                    'YEAR': event['YEAR']
                }
                
                # Process each planet
                for planet, pos in positions.items():
                    sign = self.get_sign(pos)
                    house = self.get_house(pos, ascendant)
                    planet_houses[planet] = house
                    
                    self.df.at[idx, f"{planet}_POS"] = pos
                    self.df.at[idx, f"{planet}_SIGN"] = sign
                    self.df.at[idx, f"{planet}_HOUSE"] = house
                    
                    # Calculate planetary strength
                    strength = self.calculate_planetary_strength(planet, pos, house, ascendant_sign)
                    self.df.at[idx, f"{planet}_STRENGTH"] = strength
                    
                    # Update sign distribution
                    if sign != "Unknown":
                        sign_distribution[sign][planet] += 1
                    
                    # Update house combinations
                    if house is not None:
                        house_planets[house].append(planet)
                    
                    # Store features for AI
                    event_features[f"{planet}_house"] = house if house else 'None'
                    event_features[f"{planet}_sign"] = sign if sign != "Unknown" else 'None'
                    event_features[f"{planet}_strength"] = strength
                    
                    # Track by category if available
                    if 'EVENT_CATEGORY' in self.df.columns:
                        category = event['EVENT_CATEGORY']
                        category_data['category_counts'][category] += 1
                        category_data['category_planets'][category][planet][sign] += 1
                        if house:
                            category_data['category_houses'][category][house][planet] += 1
                        category_data['category_signs'][category][sign][planet] += 1
                
                # Record aspects
                for p1, p2 in combinations(PLANET_NAMES, 2):
                    aspect = 'None'
                    for aspect_info in aspects.get(p1, []):
                        if aspect_info['planet'] == p2:
                            aspect = aspect_info['aspect']
                            aspect_patterns[(p1, p2, aspect)] += 1
                            break
                    
                    col_name = f"{p1}_{p2}_ASPECT"
                    self.df.at[idx, col_name] = aspect
                    event_features[col_name] = aspect
                
                # Identify yogas
                yogas = self.identify_yogas(positions, planet_houses)
                for yoga in yogas:
                    yoga_patterns[yoga['name']] += 1
                    event_features[f"yoga_{yoga['name']}"] = True
                
                # Record house combinations
                for house in range(1, 13):
                    if house_planets[house]:
                        combination = tuple(sorted(house_planets[house]))
                        house_combinations[house][combination] += 1
                
                feature_data.append(event_features)
                processed_count += 1
            
            except Exception as e:
                print(f"Error processing event {idx}: {e}")
                continue

        # Advanced AI Analysis
        ai_insights = {
            'clusters': [], 
            'important_features': [],
            'correlations': [],
            'temporal_patterns': [],
            'network_analysis': {},
            'category_insights': [],
            'sign_distribution_sorted': []
        }
        
        # Sort sign distribution by planet frequency
        for sign in sign_distribution:
            sorted_planets = sorted(
                sign_distribution[sign].items(),
                key=lambda x: x[1],
                reverse=True
            )
            total_sign_events = sum(sign_distribution[sign].values())
            ai_insights['sign_distribution_sorted'].append({
                'sign': sign,
                'total_events': total_sign_events,
                'percentage': round(total_sign_events / (processed_count * len(PLANET_NAMES)) * 100, 2),
                'planets': [
                    {'planet': planet, 'events': count, 'percentage_in_sign': round(count / total_sign_events * 100, 2)}
                    for planet, count in sorted_planets
                ]
            })

        if feature_data and processed_count > 1 and detailed_patterns:
            try:
                # Create feature DataFrame
                feature_df = pd.DataFrame(feature_data)
                
                # Define columns for encoding
                house_cols = [f"{planet}_house" for planet in PLANET_NAMES]
                sign_cols = [f"{planet}_sign" for planet in PLANET_NAMES]
                aspect_cols = [f"{p1}_{p2}_ASPECT" for p1, p2 in combinations(PLANET_NAMES, 2)]
                strength_cols = [f"{planet}_strength" for planet in PLANET_NAMES]
                
                # One-hot encoding pipeline
                categorical_cols = house_cols + sign_cols + aspect_cols + ['ASCENDANT_SIGN']
                yoga_cols = [col for col in feature_df.columns if col.startswith('yoga_')]
                
                # Add yoga columns if they exist
                if yoga_cols:
                    for col in yoga_cols:
                        feature_df[col] = feature_df[col].fillna(False).astype(bool).astype(int)
                
                # Prepare preprocessing pipeline
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
                        ('num', StandardScaler(), strength_cols)
                    ],
                    remainder='drop'
                )
                
                # Fit and transform features
                X = preprocessor.fit_transform(feature_df)
                
                # Get feature names
                categorical_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
                numerical_feature_names = preprocessor.named_transformers_['num'].get_feature_names_out(strength_cols)
                feature_names = np.concatenate([categorical_feature_names, numerical_feature_names])
                
                # Dimensionality reduction
                pca = PCA(n_components=min(10, X.shape[1], X.shape[0]-1))
                X_pca = pca.fit_transform(X)
                
                # Determine optimal number of clusters using silhouette scores
                max_clusters = min(8, processed_count - 1)
                silhouette_scores = []
                for n_clusters in range(2, max_clusters + 1):
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(X_pca)
                    if len(set(cluster_labels)) > 1:
                        silhouette_avg = silhouette_score(X_pca, cluster_labels)
                        silhouette_scores.append((n_clusters, silhouette_avg))
                
                # Choose the number of clusters with the highest silhouette score
                if silhouette_scores:
                    optimal_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
                else:
                    optimal_clusters = min(3, processed_count - 1)
                
                # K-Means clustering with optimal clusters
                kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_pca)
                
                # Apply t-SNE for visualization coordinates
                if processed_count >= 5:
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, processed_count/4))
                    X_tsne = tsne.fit_transform(X_pca)
                    
                    # Store coordinates for each data point
                    for i, coords in enumerate(X_tsne):
                        feature_data[i]['tsne_x'] = coords[0]
                        feature_data[i]['tsne_y'] = coords[1]
                        feature_data[i]['cluster'] = int(cluster_labels[i])
                
                # Summarize clusters
                for cluster_id in range(optimal_clusters):
                    cluster_mask = cluster_labels == cluster_id
                    cluster_size = cluster_mask.sum()
                    if cluster_size == 0:
                        continue
                    
                    cluster_percentage = (cluster_size / processed_count * 100)
                    cluster_indices = np.where(cluster_mask)[0]
                    
                    # Find dominant features in this cluster
                    cluster_df = feature_df.iloc[cluster_indices]
                    
                    # Find dominant signs, houses, and aspects
                    dominant_signs = Counter()
                    dominant_houses = Counter()
                    dominant_aspects = Counter()
                    house_planet_dist = {house: Counter() for house in range(1, 13)}
                    
                    for idx in cluster_indices:
                        data = feature_data[idx]
                        for planet in PLANET_NAMES:
                            sign = data.get(f"{planet}_sign")
                            if sign and sign != 'None':
                                dominant_signs[sign] += 1
                            
                            house = data.get(f"{planet}_house")
                            if house and house != 'None':
                                dominant_houses[house] += 1
                                house_planet_dist[house][planet] += 1
                        
                        for aspect_col in aspect_cols:
                            aspect = data.get(aspect_col)
                            if aspect and aspect != 'None':
                                dominant_aspects[aspect] += 1
                    
                    # Get top 3 dominant features for each category
                    top_signs = [sign for sign, count in dominant_signs.most_common(3)]
                    top_houses = [house for house, count in dominant_houses.most_common(3)]
                    top_aspects = [aspect for aspect, count in dominant_aspects.most_common(3)]
                    
                    # Include planet distributions for top houses
                    house_planet_summary = {
                        house: [
                            {'planet': p, 'count': c}
                            for p, c in house_planet_dist[house].most_common()
                        ]
                        for house in top_houses
                    }
                    
                    ai_insights['clusters'].append({
                        'cluster_id': cluster_id + 1,
                        'size': cluster_size,
                        'percentage': round(cluster_percentage, 2),
                        'dominant_signs': top_signs,
                        'dominant_houses': top_houses,
                        'dominant_aspects': top_aspects,
                        'house_planet_distribution': house_planet_summary,
                        'tsne_coords': {
                            'x': np.mean([feature_data[i]['tsne_x'] for i in cluster_indices]),
                            'y': np.mean([feature_data[i]['tsne_y'] for i in cluster_indices])
                        } if 'tsne_x' in feature_data[0] else None
                    })
                
                # Feature importance analysis
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X, cluster_labels)
                importances = rf.feature_importances_
                
                # Get top 10 important features
                top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
                ai_insights['important_features'] = [
                    {'feature': feature, 'importance': round(imp, 4)}
                    for feature, imp in top_features
                ]
                
                # Correlation analysis
                numerical_df = feature_df[strength_cols + ['YEAR']]
                if len(numerical_df.columns) > 1:
                    corr_matrix = numerical_df.corr()
                    top_correlations = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            col1 = corr_matrix.columns[i]
                            col2 = corr_matrix.columns[j]
                            corr = corr_matrix.iloc[i, j]
                            if abs(corr) > 0.3:
                                top_correlations.append({
                                    'feature1': col1,
                                    'feature2': col2,
                                    'correlation': round(corr, 3)
                                })
                    ai_insights['correlations'] = sorted(
                        top_correlations, 
                        key=lambda x: abs(x['correlation']), 
                        reverse=True
                    )[:10]
                
                # Network analysis of planetary aspects
                G = nx.Graph()
                for (p1, p2, aspect), count in aspect_patterns.items():
                    if aspect != 'None':
                        G.add_edge(p1, p2, weight=count, aspect=aspect)
                
                if G.edges():
                    # Calculate centrality measures
                    degree_centrality = nx.degree_centrality(G)
                    betweenness_centrality = nx.betweenness_centrality(G)
                    
                    ai_insights['network_analysis'] = {
                        'most_connected_planets': sorted(
                            degree_centrality.items(), 
                            key=lambda x: x[1], 
                            reverse=True
                        )[:3],
                        'most_bridging_planets': sorted(
                            betweenness_centrality.items(), 
                            key=lambda x: x[1], 
                            reverse=True
                        )[:3],
                        'strongest_aspects': sorted(
                            [(data['aspect'], u, v, data['weight']) for u, v, data in G.edges(data=True)],
                            key=lambda x: x[3],  # Sort by weight
                            reverse=True
                        )[:5]
                    }
                
                # Temporal patterns analysis
                if 'YEAR' in feature_df.columns:
                    yearly_counts = feature_df['YEAR'].value_counts().sort_index()
                    if len(yearly_counts) > 3:
                        peak_years = yearly_counts.nlargest(3).index.tolist()
                        ai_insights['temporal_patterns'].append({
                            'type': 'yearly_distribution',
                            'peak_years': peak_years,
                            'description': f"Peak activity in years {', '.join(map(str, peak_years))}"
                        })
                
                # Category-specific insights
                if category_data and category_data['category_counts']:
                    for category, count in category_data['category_counts'].items():
                        if count < 3:
                            continue
                            
                        # Find most common planetary positions for this category
                        top_planets = []
                        for planet, sign_counts in category_data['category_planets'][category].items():
                            if sign_counts:
                                top_sign = sign_counts.most_common(1)[0][0]
                                top_planets.append({
                                    'planet': planet,
                                    'sign': top_sign,
                                    'count': sign_counts[top_sign]
                                })
                        
                        top_planets.sort(key=lambda x: x['count'], reverse=True)
                        
                        # Find most common houses for this category
                        top_houses = []
                        for house, planet_counts in category_data['category_houses'][category].items():
                            if planet_counts:
                                total = sum(planet_counts.values())
                                top_houses.append({
                                    'house': house,
                                    'count': total
                                })
                        
                        top_houses.sort(key=lambda x: x['count'], reverse=True)
                        
                        ai_insights['category_insights'].append({
                            'category': category,
                            'count': count,
                            'percentage': round(count / processed_count * 100, 2),
                            'top_planets': top_planets[:3],
                            'top_houses': top_houses[:3]
                        })
                
                # Yoga pattern analysis
                if yoga_patterns:
                    ai_insights['yoga_patterns'] = [
                        {'yoga': yoga, 'count': count, 'percentage': round(count / processed_count * 100, 2)}
                        for yoga, count in yoga_patterns.most_common(5)
                    ]
                
            except Exception as e:
                print(f"AI analysis error: {e}")
                import traceback
                traceback.print_exc()

        # Prepare final results
        results = {
            'processed_events': processed_count,
            'total_events': len(events),
            'sign_distribution': {
                sign: {planet: count for planet, count in counter.items()}
                for sign, counter in sign_distribution.items()
            },
            'house_combinations': {
                house: {combination: count for combination, count in counter.most_common(5)}
                for house, counter in house_combinations.items()
            },
            'aspect_patterns': [
                {'planets': (p1, p2), 'aspect': aspect, 'count': count}
                for (p1, p2, aspect), count in aspect_patterns.most_common(10)
            ],
            'yoga_patterns': ai_insights.get('yoga_patterns', []),
            'ai_insights': {
                'clusters': ai_insights['clusters'],
                'important_features': ai_insights['important_features'],
                'correlations': ai_insights['correlations'],
                'temporal_patterns': ai_insights['temporal_patterns'],
                'network_analysis': ai_insights['network_analysis'],
                'category_insights': ai_insights['category_insights'],
                'sign_distribution_sorted': ai_insights['sign_distribution_sorted']
            },
            'category_data': {
                'category_counts': dict(category_data['category_counts']),
                'category_planets': {
                    cat: {p: dict(counter) for p, counter in planets.items()}
                    for cat, planets in category_data['category_planets'].items()
                },
                'category_houses': {
                    cat: {h: dict(counter) for h, counter in houses.items()}
                    for cat, houses in category_data['category_houses'].items()
                },
                'category_signs': {
                    cat: {s: dict(counter) for s, counter in signs.items()}
                    for cat, signs in category_data['category_signs'].items()
                }
            } if category_data else {},
            'feature_data': feature_data,
            'updated_dataframe': self.df.to_dict(orient='records')
        }

        return results