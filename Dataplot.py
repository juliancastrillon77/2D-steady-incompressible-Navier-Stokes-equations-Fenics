# Julian Castrillon
# CFD - Spring 2022
# Data plot

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

Forces = np.loadtxt('ForcesEllipseQ1Q1.txt', dtype='float', delimiter=' ')

SR = 100 # Sample rate
Duration = 5 # Duration of the sample

StartTime = 25*SR
Drag = Forces[StartTime:-1,0]
Lift = Forces[StartTime:-1,1]
Cd = Forces[StartTime:-1,2]
Cl = Forces[StartTime:-1,3]

time = np.linspace(0,np.size(Drag,0),np.size(Drag,0),endpoint=False)*(1/SR)

plt.figure(1)
plt.grid()
plt.title('Drag')
plt.ylabel('Drag')
plt.xlabel('Time')
plt.plot(time,Drag,'-',color='#A2142F')

plt.figure(2)
plt.grid()
plt.title('Lift')
plt.ylabel('Lift')
plt.xlabel('Time')
plt.plot(time,Lift,'-',color='#4DBEEE')

plt.figure(3)
plt.grid()
plt.title('Drag Coefficient')
plt.ylabel('Cd')
plt.xlabel('Time')
plt.plot(time,Cd,'-',color='#D95319')

plt.figure(4)
plt.grid()
plt.title('Lift Coefficient')
plt.ylabel('Cl')
plt.xlabel('Time')
plt.plot(time,Cl,'-',color='#77AC30')

## Fourier transform

DragFt = rfft(Drag)
LiftFt = rfft(Lift)
CdFt   = rfft(Cd)
ClFt   = rfft(Cl)
xf = rfftfreq(SR*Duration, 1/SR)

plt.figure(5)
plt.grid()
plt.title('Fourier transform')
plt.ylabel('Power')
plt.xlabel('Frequencies')
plt.plot(xf[0:-1], np.abs(LiftFt),color='#4DBEEE')

S = np.abs(LiftFt)
maxi = np.argmax(S)
St = xf[maxi+1]*0.2/1.5 # Strouhal Number

print('\n')
print('Strouhal Number: %.5f \n\n' % St, '\n\n')



#St = np.max(abs_Lift)*0.2/1.5
#plt.plot(xf[0:-1], np.abs(CdFt),color='#D95319')
#plt.plot(xf[0:-1], np.abs(ClFt),color='#77AC30')

