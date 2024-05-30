import numpy as np
import math
import matplotlib.pyplot as plt

# Constants
earth_radius = 6378000  # in meters
r_loss = 0.75
w0 = 2.5 * 10 ** (-2)
lam = 810 * 10 ** (-9)
trans_atm_z = 0.5
r_noise = 0.5
dlam = 10 ** (-9)
omega = 100 * 10 ** (-6)
dT = 10 ** (-9)
R_source = 10**9
T = 24  # in hours
LR = math.pi * w0**2 / lam
pairs = [
    [2, 10], [3, 10], [4, 5], [5, 5], [6, 5], [7, 5], [8, 5], [9, 5], [4, 6], [5, 6], [6, 6], [7, 6], [8, 6], [9, 6],
    [4, 8], [9, 7], [8, 7], [7, 7], [6, 7], [5, 7], [4, 7], [8, 10], [9, 10], [8, 11], [10, 10], [4, 13], [5, 13], [7, 13],
    [5, 8], [6, 8], [7, 8], [8, 8], [9, 8], [8, 9], [9, 9], [7, 14], [7, 15], [10, 14], [10, 15], [15, 15], [16, 16], [20, 20]
]

# Functions as defined previously
def rotation(spherical0, t):
    theta0 = spherical0[1]
    phi0 = spherical0[2]

    theta = theta0 + 2 * math.pi / T * t
    phi = (phi0 + 2 * math.pi / T * t) % (2 * math.pi)

    if theta > math.pi:
        phi = (phi + math.pi) % (2 * math.pi)
        theta = theta % math.pi

    return np.array([spherical0[0], theta, phi])

def spherical_to_cartesian(spherical):
    x = spherical[0] * math.sin(spherical[1]) * math.cos(spherical[2])
    y = spherical[0] * math.sin(spherical[1]) * math.sin(spherical[2])
    z = spherical[0] * math.cos(spherical[1])
    return np.array([x, y, z])

def w(L):
    # print(w0 * math.sqrt(1 + (L / LR) ** 2))
    return w0 * math.sqrt(1 + (L / LR) ** 2)

def trans_fs(L):
  # print(1 - math.exp(-2 * r_loss ** 2 / (w(L) ** 2)))
  return 1 - math.exp(-2 * r_loss ** 2 / (w(L) ** 2))

def zeta(L, h):
    return math.acos(h / L - (L ** 2 - h ** 2) / (2 * earth_radius * L))

def trans_atm(L, h):
    zet = zeta(L, h)
    if zet >= math.pi / 2 or zet <= -math.pi / 2:
        return 0
    else:
        # print(trans_atm_z ** (1 / math.cos(zet)))
        return trans_atm_z ** (1 / math.cos(zet))

def compute_L(i, j, t, h, d, pair):
    NR = pair[0]
    NS = pair[1]
    d_theta = 2 * math.pi / NS
    d_phi = math.pi / NR
    nr = (i - 1) % NS
    ns = (i - 1 - nr) / NS
    spherical0_i = np.array([earth_radius + h, d_theta * ns, d_phi * nr])
    spherical_j = np.array([earth_radius, math.pi / 2, d / earth_radius * (j - 1 / 2)])

    spherical_i = rotation(spherical0_i, t)

    cartesian_i = spherical_to_cartesian(spherical_i)
    cartesian_j = spherical_to_cartesian(spherical_j)
    # print( np.linalg.norm(cartesian_i - cartesian_j))
    return np.linalg.norm(cartesian_i - cartesian_j)

# def compute_L(i, j, t, h, d, pair):
#     NR = pair[0]
#     NS = pair[1]
#     # Determine the ring and position within the ring
#     r = i // NS
#     pos = i % NS

#     # Calculate the latitude and longitude of the satellite
#     latitude_ring_spacing = 180 / NR
#     latitude = (r - NR/2) * latitude_ring_spacing
#     longitude = pos * (360 / NS)

#     # Convert latitude and longitude to radians
#     latitude_rad = math.radians(latitude)
#     longitude_rad = math.radians(longitude)

#     # Calculate the satellite's Cartesian coordinates
#     satellite_radius = earth_radius + h
#     x_satellite = satellite_radius * math.cos(latitude_rad) * math.cos(longitude_rad)
#     y_satellite = satellite_radius * math.cos(latitude_rad) * math.sin(longitude_rad)
#     z_satellite = satellite_radius * math.sin(latitude_rad)

#     # Calculate the ground base longitude
#     longitude_base = d / 2 if j == 2 else -d / 2
#     longitude_base_rad = math.radians(longitude_base)

#     # Ground base is on the equator, so latitude is 0
#     x_base = earth_radius * math.cos(longitude_base_rad)
#     y_base = earth_radius * math.sin(longitude_base_rad)
#     z_base = 0

#     # Calculate the distance using the 3D distance formula
#     distance = math.sqrt((x_satellite - x_base)**2 + (y_satellite - y_base)**2 + (z_satellite - z_base)**2)

#     print(distance)
#     return distance

def trans_sg(i, j, t, h, pair, d):
    L = compute_L(i, j, t, h, d, pair)
    # print(f"Satellite {i}, Ground {j}: L = {L}")
    # print((trans_fs(L) * trans_atm(L, h))**2)
    fs = trans_fs(L)
    atm = trans_atm(L, h)

    # print(f"Satellite {i}, Ground {j}: trans_fs = {fs}, trans_atm = {atm}")

    return fs * atm

def trans_tot(i, j1, j2, t, h, pair, d):
    if i == 0:
        return 0
    return trans_sg(i, j1, t, h, pair, d) * trans_sg(i, j2, t, h, pair, d)

# def satellite_range(i, j1, j2, t, h, pair, d):
#     tt = trans_tot(i, j1, j2, t, h, pair, d)
#     if tt == 0:
#         return 0
#     elif -10 * math.log10(tt) < 90:
#         return 0
#     return 1

def satellite_range(i, j1, j2, t, h, pair, d):
    tt = trans_tot(i, j1, j2, t, h, pair, d)
    # print(f"Satellite {i}: trans_tot = {tt}")

    if tt == 0:
        # print(f"Satellite {i}: trans_tot is 0, returning 0")
        return 0
    elif -10 * math.log10(tt) < 90:
        # print(f"Satellite {i}: Loss {-10 * math.log10(tt)} is less than 90, returning 0")
        return 0
    return 1

def st(j1, j2, pair, t, h, d):
    num = pair[0] * pair[1]
    satelite = 0
    loss = 0
    for i in range(1, num + 1):
        if satellite_range(i, j1, j2, t, h, pair, d) == 1:
            new_loss = -10 * math.log10(trans_tot(i, j1, j2, t, h, pair, d))
            if loss == 0 or new_loss < loss:
                satelite = i
                loss = new_loss
    return satelite

def average_rate(j1, j2, T, pair, h, d):
    sum = 0
    for t in range(1, T + 1):
      stj = st(j1, j2, pair, t, h, d)
      if satellite_range(stj, j1, j2, t, h, pair, d) == 0:
          continue
      sum += R_source * trans_tot(stj, j1, j2, t, h, pair, d)
    #   print(trans_tot(stj, j1, j2, t, h, pair, d))
    # print("next")
    return sum /T

def optimal_rate(j1, j2, T, h, d):
    # max_rate = 0
    # for pair in pairs:
    #     rate = average_rate(j1, j2, T, pair, h, d)
    #     if max_rate < rate:
    #         max_rate = rate
    # return max_rate
    pair = [20, 20]
    return average_rate(j1, j2, T, pair, h, d)

# Simulation parameters
d_meters = 500 * 1000  # Convert distance to meters
h_values_meters = range(200 * 1000, 10001 * 1000, 200 * 1000)  # Heights in meters from 200 to 10000 with increment of 200

# Calculate optimal rates
optimal_rates = [optimal_rate(1, 2, T, h, d_meters) for h in h_values_meters]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot([h / 1000 for h in h_values_meters], optimal_rates, marker='o')  # Convert height back to kilometers for plotting
plt.xlabel('Height (km)')
plt.ylabel('Optimal Rate')
plt.title('Optimal Rate vs Height for d = 500 km')
plt.grid(True)
plt.show()

# Simulation parameters
d_meters = 500 * 1000  # Convert distance to meters
h_values_meters = range(200 * 1000, 10001 * 1000, 200 * 1000)  # Heights in meters from 200 to 10000 with increment of 200

# pair = pairs[19]
# Calculate optimal rates
# average_rates = [average_rate(1, 2, T, pair, h, d_meters) for h in h_values_meters]

for pair in pairs:
  for t in range(1, T+1):
    sat = st(1, 2, pair, t, 500000, 500000)
    print(sat)
