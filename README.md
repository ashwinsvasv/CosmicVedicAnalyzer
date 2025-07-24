# 🌌 Cosmic Vedic Astrology Analyzer 🌠

**Unlock celestial insights from event data using ancient Vedic astrology combined with modern data science**

![1](/api/static/1.png)
## 📌 Overview

This project analyzes historical or contemporary events through the lens of Vedic astrology, revealing hidden planetary patterns and cosmic influences. It processes event data (like earthquakes, financial events, or personal milestones) and performs sophisticated astrological calculations combined with machine learning analysis.

## Give us a ⭐️ if you find this project helpful!  

If you like this project, please consider giving it a star ⭐️ on GitHub. Your support motivates me to keep improving it!  

<p align="center">
  <a href="https://buymeacoffee.com/ishanoshada">
    <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" height="50" alt="Buy Me a Coffee">
  </a>
</p>


## 🔍 Key Features

![1](/api/static/5.png)
### 📊 Data Processing
- **Flexible CSV parsing** - Handles multiple date/time formats automatically
- **Smart column detection** - Identifies date, location, magnitude columns
- **Data normalization** - Converts diverse formats into standardized astrological inputs

![1](/api/static/2.png)
### ♉ Vedic Astrology Analysis
- **Planetary Positions**: Calculates positions of all 9 Vedic planets (including Rahu/Ketu)
- **House System**: 12-house Bhāva system with accurate cusp calculations
- **Aspects (Drishti)**: Full Vedic aspect patterns including special planetary aspects
- **Yogas**: Detects significant planetary combinations (Raj Yoga, Gajakesari Yoga, etc.)
- **Planetary Strength**: Shadbala-inspired strength calculations

### 🤖 Advanced AI Analysis
- **Event Clustering**: Groups similar astrological patterns using K-Means and t-SNE
- **Feature Importance**: Identifies most influential planetary positions using Random Forests
- **Network Analysis**: Visualizes planetary relationship networks
- **Temporal Patterns**: Detects cyclical patterns in event occurrences

## 🛠 Technical Implementation

### Backend (Python/Flask)
- **Astronomical Calculations**: 
  - Uses PyEphem for precise planetary positions
  - Implements Lahiri Ayanamsa for sidereal zodiac
  - Calculates special lunar nodes (Rahu/Ketu)

- **Machine Learning**:
  - **Dimensionality Reduction**: PCA + t-SNE for visualization
  - **Clustering**: Optimal K-Means clustering with silhouette scoring
  - **Feature Analysis**: Random Forest feature importance
  - **Network Analysis**: NetworkX for planetary aspect graphs

![1](/api/static/3.png)
### Frontend
- **Interactive Visualizations**: Chart.js for planetary distributions and patterns
- **Cosmic UI**: Particle.js animated background with responsive design
- **Animated Transitions**: AOS library for smooth scrolling effects

## 📂 Sample Datasets Included

1. **Earthquakes (1995-2023)**
   - 28 years of global seismic activity
   - Contains magnitude, depth, and location data
   - Demonstrates planetary patterns in natural disasters

2. **Historical Tsunamis**
   - Major tsunami events with impact data
   - Shows lunar nodal (Rahu/Ketu) correlations

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
```bash
# Clone repository
git clone https://github.com/ishanoshada/CosmicVedicAnalyzer.git
cd CosmicVedicAnalyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
python api/app.py
```

Visit `http://localhost:5000` in your browser after starting the application.

## 📝 How to Use

1. **Prepare Your Data**:
   - Create CSV with events (date/time + location required)
   - Example columns: "2020-01-01 14:30", 34.0522, -118.2437 (LA coordinates)

2. **Upload & Analyze**:
   - Drag-and-drop your CSV file
   - System auto-detects columns and formats

3. **Interpret Results**:
   - View planetary position distributions
   - Examine significant yogas and aspects
   - Explore AI-generated clusters and patterns


![1](/api/static/4.png)
## 🧠 Astrological Methodology

### Core Calculations
1. **Sidereal Zodiac**: Uses fixed-star based 27° nakshatras
2. **House System**: Whole-sign houses from Ascendant
3. **Planetary Aspects**:
   - Standard aspects (conjunction, sextile, square, etc.)
   - Special Vedic aspects (Mars 4/8, Jupiter 5/9, etc.)
4. **Planetary Strength**:
   - Sign placement (exaltation/debilitation)
   - House position (kendras/trikonas)
   - Aspect relationships

### Yoga Detection
- Identifies 50+ classical yoga combinations
- Weighted by traditional astrological texts
- Contextualized to event data patterns

## 🤝 Contributing

We welcome contributions! Please see our [Contribution Guidelines](CONTRIBUTING.md) for details.

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

## 🌟 Acknowledgments

- The Jyotish (Vedic astrology) tradition
- Python astronomical community
- Open-source data science ecosystem

---

<div align="center">
  <p>"As above, so below" - Hermes Trismegistus</p>
  <p>
    <a href="https://github.com/ishanoshada">GitHub</a> • 
 • 
    <a href="mailto:ic31908@gmail.com">Contact</a>
  </p>
</div>


![Views](https://dynamic-repo-badges.vercel.app/svg/count/6/Repository%20Views/astroana)
