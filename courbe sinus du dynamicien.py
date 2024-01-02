from math import pi,atan,tan,sin,cos, floor
import matplotlib.pyplot as plt
import numpy as np
from math import inf
from scipy import optimize
from random import random
f = lambda x: sin(x)
fp = lambda  x: cos(x)
eps =0.00001
space= 0.0003


g=9.81
L1,L2=[],[] # pour avoir un truc graphique

def signefun(x):
    if x<0:
        return -1
    if x==0:
        return 0
    return 1

def dicho(a,b,f,eps):
    p,q=a,b
    while abs(f(p)-f(q))>eps*0.0001:
        m=(p+q)/2
        if f(m)*f(q)>0:
            q=m
        else:
            p=m
    return m



def trouve_sol_proche(f,x0,signe,eps): # on suppose la fonction positive car on reste toujours au dessus de la courbe du sinus
    xmin=x0
    a,x=f(x0),x0
    while a>0:
        x+=signe*eps
        a=f(x)
    signebis=signefun(x-x0)
    b=dicho(x-100*eps,x+100*eps,f,eps)
    N=floor(abs(x-x0)/space)
    X=[x0 + space*k*signebis for k in range(N)]
    return b,X,[f(x) + sin(x) for x in X]



def evolue(xn,deltan,v0,signeprec):#signeprec est le signe d'avancement du précédent tir, # E EST L'ENERGIE MECANIQUE DU SYSTEME, on stockera plus tard seulement v0**2
    thetaf=atan(cos(xn))
    deltabis = pi-deltan+2*thetaf
    if thetaf>=0:
        if signeprec>=0:
            if deltan%(2*pi)-2*pi<=thetaf+pi/2:
                signe=-1
            else:
                signe=1
        else:
            signe=-1
    else:
        if signeprec<=0:
            if deltan<=0 or deltan<=thetaf+pi/2:
                signe=-1
            else:
                signe=1
        else:
            signe=1
    xn_1,A,B=trouve_sol_proche(lambda x: -g*((x-xn)**2)/(2*v0*v0*cos(deltabis)*cos(deltabis))+tan(deltabis)*(x-xn)+sin(xn)-sin(x),xn+10*signe*eps,signe,eps) # on commence à xn + signe*eps pour pouvoir commencer forcément dans le positif après avoir bougé un petit peu 
    if signe<0:
        deltan_1=atan(tan(deltabis)-g*(xn_1-xn)/(v0*v0*cos(deltabis)*cos(deltabis)))
    else:
        deltan_1=pi+atan(tan(deltabis)-g*(xn_1-xn)/(v0*v0*cos(deltabis)*cos(deltabis)))
    v0n_1=(v0*v0+2*g*(sin(xn)-sin(xn_1)))**0.5 # ce sont les altitudes de la balle
    return xn_1,deltan_1,v0n_1,signe,deltabis,A,B


def graphe(n,x0,v0,delta0,signeprec): # n le nombre d'itérations, N les pas utilisées pour le tracé
    x,delta,v,signe=x0,delta0,v0,signeprec
    xmin,xmax=inf,-inf
    liste1,liste2=[],[]
    for _ in range(n):
        x1,delta1,v1,signeprov,deltabis,A,B=evolue(x,delta,v,signe)
        a,b=list(A),list(B)
        liste1+=a
        liste2+=b
        xmin=min(xmin,x1)
        xmax=max(xmax,x1)
        x,delta,v,signe=x1,delta1,v1,signeprov
    return x,delta,v,liste1,liste2,xmin,xmax



_,_,_,X,Y,xmin,xmax=graphe(15,-1,6.9,pi/2.2,1)
# graphe(20,-0.5,6.91,pi/3,1)




import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig,ax= plt.subplots() # initialise la figure
ax.grid()
line, = plt.plot([],[]) 
plt.xlim(xmin-2,xmax+2)
plt.ylim(-2,2)

H=np.linspace(xmin-2,xmax+2,floor(200000*(xmax-xmin)))
plt.plot(H,[sin(y) for y in H],c='blue')
# fonction à définir quand blit=True, toujours pas compris cette remarque

# crée l'arrière de l'animation qui sera présent sur chaque image
def init():
    line.set_data([],[])
    return line,

def animate(i): # fonction majeure
    line.set_data(X[:250*i],Y[:250*i])
    return line,
 
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=10000, blit=True, interval=10, repeat=True)

plt.show()