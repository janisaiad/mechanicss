from math import cos,sin,pi,tan
from random import random


l0=2

def rk4vect_temp(F,x0,a,h,n):
    L1,L2,z,x=[a[0]],[a[1]],a,x0
    for _ in range(1,n):
        x+=h
        p,q=z[0],z[1]
        k1,kk1=F(p,q,x)
        k2,kk2=F(p+h/2*k1,q+h/2*kk1,x+h/2)
        k3,kk3=F(p+h/2*k2,q+h/2*kk2,x+h/2)
        k4,kk4=F(p+h*k3,q+h*kk3,x+h)
        z=p+h/6*(k1+2*k2+2*k3+k4),q+h/6*(kk1+2*kk2+2*kk3+kk4)
        L1.append(z[0])
        L2.append(z[1])
    return L1,L2

def afficher5_temp(F,c,a,h,n):
    import matplotlib.pyplot as plt
    import numpy as np
    ax=plt.subplots()[1]
    ax.set(xlabel='x', ylabel='y')
    ax.grid()
    p=np.linspace(c[0],c[0]+n*h,n)
    q=np.linspace(c[1],c[1]+n*h,n)
    b1,b2=rk4vect_temp(F,c[0],a,h,n)
    plt.plot(p,b1)
    plt.plot(q,b2,color='black')
    plt.show()
    return b1,b2


def afficher6_temp(F,c,a,h,n): # portrait de phase simple
    import matplotlib.pyplot as plt
    import numpy as np
    ax=plt.subplots()[1]
    ax.set(xlabel='x', ylabel='y')
    ax.grid()
    b1,b2=rk4vect_temp(F,c[0],a,h,n)
    plt.plot(b1,b2,color='black')
    plt.show()
    return b1,b2


# paramètres : beaux : a=4 ou a = 0.6, q=9, T=0.2, frottements = 0
# a grand permet de rendre moins sensible aux perturbations sinusoidales
a=0 # position d'équilibre
q=2# amplitude
T=0.5  # période
frottements=0

# PTDRRRRRR VRAIMENT ALEATOIRE WSH
def omega(t):
    return a-2*q*cos(2*pi*t/T)*tan(random()*pi/2-0.1)

def fonction(x,y,t): # la constante sur le t² est un paramètre de l'équation
    return y,-x*omega(t)**2-frottements*y


# afficher5_temp(fonction,(1,1),(1,-1),0.00001,1000000)


# afficher6_temp(fonction,(1,1),(1,-1),0.00001,1000000)

# visuel 



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

X,Y=afficher5_temp(fonction,(1,1),(1,-1),0.00001,5000000) 

fig,ax= plt.subplots() # initialise la figure
ax.grid()
line, = plt.plot([],[]) 
plt.xlim(-5,5)
plt.ylim(-4,4)


# crée l'arrière de l'animation qui sera présent sur chaque image
def init():
    line.set_data([],[])
    return line,

def animate(i):
    line.set_data([l0*cos(x) for x in X[250*i:500*i]]+[0],[ l0*sin(x) for x in X[250*i:500*i]]+[0])
    return line,
 
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=2000, blit=True, interval=10, repeat=False)

plt.show()