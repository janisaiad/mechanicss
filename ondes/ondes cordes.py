T,mu=1,0.1
v = (T/mu)**0.5
g=9.81
frott=3

from math import exp,sin,pi,cos,tanh,cosh

from numpy import sinc



def évolue(M,Te,Xe):
    L=[0]
    N=len(M[0])
    for k in range(1,N-1):
        y=(M[-1][k-1]+M[-1][k+1]-2*M[-1][k])*(Te*v/Xe)**2+2*M[-1][k]-M[-2][k]-g*Te*Te
        L.append(y)
    L.append(0)
    M.append(L)




def évolue_sans0(M,Te,Xe):
    L=[0]
    N=len(M[0])
    for k in range(1,N-1):
        y=(M[-1][k-1]+M[-1][k+1]-2*M[-1][k])*(Te*v/Xe)**2+2*M[-1][k]-M[-2][k]-g*Te*Te-frott*(M[-1][k]-M[-2][k])*Te*Te
        L.append(y)
    L.append(y)
    M.append(L)



def init(f,f2,N): #fp la dérivée
    return [[f(k/N) for k in range(N)],[f2(k/N) for k in range(N)]]



def corde(f,f2,N,Te,Xe,ite,s): # on aura ite+2 lignes dans la matrice
    M=init(f,f2,N)
    if s==0:
        
        for _ in range(ite):
            évolue(M,Te,Xe)
    else:
        for _ in range(ite):
            évolue_sans0(M,Te,Xe)
    return M
    


# animation à effectuer

N=100
Te=0.0001
Xe=1/N

def f1(x):
    return 0.1*cosh(x-0.5)-0.1*cosh(0.5)+0.2*exp(-20.99*x)*sin(10*pi*x)

def f2(x):
    return 0.1*cosh(x-0.5)-0.1*cosh(0.5)+0.2*exp(-21*x)*sin(10*pi*x)

ite = 20000

import matplotlib.pyplot as plt
import matplotlib.animation as animation


M=corde(f1,f2,N,Te,Xe,ite,0)
L=[k/N for k in range(N)]
fig,ax= plt.subplots() # initialise la figure
ax.grid()
line, = plt.plot([],[]) 
plt.xlim(-0.2,1.2)
plt.ylim(-1,1)
# fonction à définir quand blit=True, toujours pas compris cette remarque

# crée l'arrière de l'animation qui sera présent sur chaque image
def init():
    line.set_data([],[])
    return line,


def animate(i): # fonction majeure
    line.set_data(L,M[40*i])
    return line,
 
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=100000, blit=True, interval=20, repeat=True)

plt.show()

# rendre tout ça efficace avec numpy et étudier tout ça, les perturbations seuils etc ..




# on fait la version corde en l'air
# ensuite on fera des simulations en 2d pour des surfaces quelconques (euh voir transformation conforme)
#  puis on pourra adapter à des plans inclinés et des tensions différentes
