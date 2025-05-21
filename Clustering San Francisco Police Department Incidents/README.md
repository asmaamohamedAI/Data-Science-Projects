
# Clustering San Francisco Police Department Incidents

This project demonstrates geospatial data analysis using Python and the `folium` library. It visualizes and clusters police department incidents in San Francisco on interactive maps.

## Features

- Display world and US maps using `folium`.
- Explore different map styles, such as Stamen Toner and Stamen Terrain.
- Visualize San Francisco police incidents on a map using circle markers.
- Add pop-up text labels and markers to the map.
- Group incident markers into clusters for better visualization.

## Requirements

- Python 3.x
- Required Python libraries:
  - `folium`
  - `pandas`

## Installation

1. Clone this repository or download the project files.
2. Install the required Python libraries:
   ```bash
   pip install folium pandas
   ```

## Usage

1. Open the Jupyter Notebook file `Clustering San Francisco Police Department Incidents_Application1.ipynb`.
2. Run the cells sequentially to:
   - Load the dataset.
   - Visualize incidents on a map.
   - Experiment with different map styles and clustering techniques.

## Dataset

The dataset used in this project is publicly available and contains information about police department incidents in San Francisco for the year 2016. It includes the following features:
- Incident number, category, description, date, time, location, and more.

The dataset is loaded directly from a URL:
```
https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DV0101EN/labs/Data_Files/Police_Department_Incidents_-_Previous_Year__2016_.csv
```

## Examples

### Displaying a Map of San Francisco
The project creates an interactive map centered on San Francisco:
```python
latitude = 37.77
longitude = -122.42
sanfran_map = folium.Map(location=[latitude, longitude], zoom_start=12)
sanfran_map
```

### Clustering Incidents
Markers for incidents are grouped into clusters for better visualization:
```python
from folium import plugins
incidents = plugins.MarkerCluster()
for lat, lng, label in zip(df_incidents.Y, df_incidents.X, df_incidents.Category):
    incidents.add_child(folium.Marker([lat, lng], popup=label))
sanfran_map.add_child(incidents)
```

## Map Styles
- **Stamen Toner**: High-contrast black-and-white maps.
- **Stamen Terrain**: Maps with hill shading and natural vegetation colors.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset provided by Cognitive Class.
- Visualization powered by the `folium` library.
```




