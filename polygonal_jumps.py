import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numpy.fft import fft,ifft

# Constants
c1 = 1.13
c2 = 1
c3 = 1
c4 = 0.3
g = 9.82 

mu = 1.61*10**(-2) # I think this is for ethylene glycol, or something. Can be changed
rho = 1.1132 * 1000
nu = mu/rho
h_i = ? *(1/1000) # Some units, are changed
h_o = ? *(1/1000) 
R = ? *10**(-2) 
Q = ? *(1/1000000) 
w = ?
sigma = ? *10**(-3) 
sigma = 0

Bo = h_o/np.sqrt(sigma/(rho*g)) 
PI1 = (g*h_i**2 * h_o**2)/(2*c1 * nu * Q) 
PI2 = (2*c2 * nu * Q)/(g*R**2 * h_o**2) 
PI3 = (2*c3*nu*Q)/(g*h_i*h_o**3)
phi = (2*np.pi)**2 * PI1**(3/2) * PI2**(-1/2)

# Functions
def f(t, r): # The equations depend on whether it is the rotational model, or the normal etc.
    delta, ep = r
    fd = 1 # delta diff eq
    fe = 1 # xi diff eq
    return fd, fe

def ev1(t, r): # An event, used for event solving the model
    delta, ep = r
    return # fd, fra før

# initial conditions
T = []
vals = []
for deltas in np.linspace(0.01, 1, 100):
    for eps in np.linspace(0.01, 1, 100):
        delta_0 = deltas
        ep_0 = eps
        sol = solve_ivp(f, (0, 1000), (delta_0, ep_0), t_eval=np.linspace(0, 1000, 50000), events=[ev1])
        try:
            Ts = -2*(sol.t_events[0][0]-sol.t_events[0][1])
            T.append(Ts)
            vals.append([delta_0, ep_0])
        except:
            T.append(1000)
            vals.append([delta_0, ep_0])
            pass
        
T = np.array(T)
vals = np.array(vals)
N = 2*np.pi/np.median(T)
temp = T-np.pi/N
i = list(temp).index(min(temp))
sol = solve_ivp(f, (0, 2*np.pi), (vals[i, 0], vals[i, 1]), t_eval=np.linspace(0, 2*np.pi, 100000), events=[ev1])


delt, eps = sol.y
theta = sol.t
y_e = np.array(sol.y_events)
d_e = y_e[:, :, 0]
plt.plot(sol.t, delt)
plt.plot(sol.t_events, d_e, "ro")
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
#theta = np.linspace(0, 2*np.pi, len(list(delt)))
ax.plot(theta, 1-delt, label="Hydraulic Jump")
#ax.plot(theta, PI1/delt)
#ax.plot(theta, eps)
plt.title("Numerical solution of the model")
#plt.yticks([])
plt.legend()
#plt.savefig("fig_name.png")
plt.show()

y = fft(delt)
x = np.linspace(0, 1000, len(y))
plt.figure(figsize=(12, 8))
plt.plot(x, abs(y))
plt.xlim(0.01, 0.8)
plt.ylim(0, 5000)
plt.title(rf"Fourier transform of solution for $\omega = ${w}")
#plt.xticks([])
#plt.yticks([])
plt.xlabel("Frequencies [a. u.]")
plt.ylabel("Amplitudes [a. u.]")
plt.show()




