import matplotlib.pyplot as plt
import numpy as np


import matplotlib.animation as animation

a,b =3,3

def evolue(x,y,alpha,sens,a,b):
    sensbis=sens
    if sens>=0:
        if alpha*(a-x)+y>b:
            xx=(b-y)/alpha+x
            yy=b
        elif alpha*(a-x)+y>0:
            sensbis=-1
            xx=a
            yy=alpha*(a-x)+y
        else:
            yy=0
            xx=-y/alpha+x
    else:
        if alpha*(-x)+y>b:
            xx=(b-y)/alpha+x
            yy=b
        elif alpha*(-x)+y>0:
            sensbis=1
            xx=0
            yy=alpha*(-x)+y
        else:
            yy=0
            xx=-y/alpha+x
    return xx,yy,-alpha,sensbis


def afficherbillard(x0,y0,sens0,alpha0,a,b,N,intervalle):
    x,y,sens,alpha=x0,y0,sens0,alpha0
    X,Y=[],[]
    for _ in range(N):
        xbis,ybis,alpha,sens=evolue(x,y,alpha,sens,a,b)
        X+=[x+(xbis-x)*k/(intervalle-1) for k in range(intervalle)]
        Y+=[y+(ybis-y)*k/(intervalle-1) for k in range(intervalle)]
        x,y=xbis,ybis
    return X,Y



X,Y=afficherbillard(0.253,0,1,0.47382,a,b,300,100)

fig,ax= plt.subplots() # initialise la figure
ax.grid()
line, = plt.plot([],[]) 
plt.xlim(-1,a+1)
plt.ylim(-1,b+1)
plt.plot([0,a,a,0,0],[0,0,b,b,0],color="black")
# fonction à définir quand blit=True, toujours pas compris cette remarque

# crée l'arrière de l'animation qui sera présent sur chaque image
def init():
    line.set_data([],[])
    return line,

def animate(i): # fonction majeure
    line.set_data(X[:10*i],Y[:10*i])
    return line,
 
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=1000, blit=True, interval=10, repeat=True)

plt.show()