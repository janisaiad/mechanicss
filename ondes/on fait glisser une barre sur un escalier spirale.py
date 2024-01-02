
import numpy as np
from math import pi,cos,sin,tan
import matplotlib.pyplot as plt
import matplotlib.animation as animation



# faire n trucs pour rigoler
L,h,g,m=1,0.1,9.81,1

R1,R2=10,1


# forçage ensuite

# phi, theta


def rk4vect_dim4(F,vectinit,h,n):
    L,z=np.zeros((n,5)),vectinit
    L[0,:]=z
    for j in range(1,n):
        k1=F(z)
        k2=F(z+k1*h/2)
        k3=F(z+k2*h/2)
        k4=F(z+k3*h)
        z=z+(k1+2*k2+2*k3+k4)*h/6
        L[j]=z
    return L

def afficher_positionsvitesse(F,a,h,n): # graphe des fonctions positions et vitesse
    import matplotlib.pyplot as plt
    import numpy as np
    from math import cos,sin
    ax=plt.subplots()[1]
    ax.set(xlabel='x', ylabel='y')
    ax.grid()
    L=rk4vect_dim4(F,a,h,n)
    plt.plot(L[:,0],L[:,1],color='green')
    plt.plot(L[:,0],L[:,3],color='blue')
    plt.show()
    return L[:,0],L[:,1],L[:,3] # t, theta, phi



def evolue(z):
    theta,thetapoint,phi,phipoint=z[1],z[2],z[3],z[4]
    return np.array([1,z[2],thetapoint*(h*phipoint*cos(phi)+R2*phi*phipoint*sin(phi))/(R1-(h+R2)*sin(phi)+R2*phi*cos(phi)), z[3],(-tan(phi)*(g-R2*phi*phipoint*sin(phi))+R2*phi*phipoint*phipoint*cos(phi)-h*phipoint*phipoint*sin(phi))/(-h*cos(phi)-R2*phi*sin(phi)+tan(phi)*h*sin(phi)-R2*phi*cos(phi))])


T,theta,phi=afficher_positionsvitesse(evolue,np.array([0,0.1,0,0.1,0]),0.00001,200000)

"""

fig,ax= plt.subplots() # initialise la figure
ax.grid()
line, = plt.plot([],[]) 
plt.xlim(-5,5)
plt.ylim(-4,4)

# fonction à définir quand blit=True

# crée l'arrière de l'animation qui sera présent sur chaque image
def init():
    line.set_data([],[])
    return line,

def animate(i):
    line.set_data()
    return line,
 
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=1000, blit=True, interval=10, repeat=False)
plt.show()

"""



