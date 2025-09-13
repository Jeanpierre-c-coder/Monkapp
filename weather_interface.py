# Import Meteostat library and dependencies
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily

# Set time period
start = datetime(2025, 1, 1)
end = datetime(2025, 9, 14)

# Create Point for Paris, France
from meteostat import Point

# Latitude, Longitude, Altitude (mètres)
location = Point(48.8566, 2.3522, 35)

print(location)
# Affichage des résultats (à activer si vous avez les variables times et temperatures)
# plt.figure(figsize=(12, 6))
# plt.plot(times, [temp[0] for temp in temperatures], label='T_int (Intérieur)', color='blue')
# plt.plot(times, [temp[1] for temp in temperatures], label='T_ext (Extérieur)', color='orange')
# plt.plot(times, [temp[2] for temp in temperatures], label='T_wall (Mur)', color='green')
# plt.xlabel('Heure de la journée')
# plt.ylabel('Température (°C)')
# plt.title('Simulation thermique du bâtiment sur 24h')
# plt.legend()
# plt.grid()
# plt.show()          

# Get daily data for 2025 for Paris, France
data = Daily(location, start, end)
data = data.fetch()

# Plot line chart including average, minimum and maximum temperature
data.plot(y=['tavg', 'tmin', 'tmax'])
plt.show()