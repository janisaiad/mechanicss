#  on a un billard avec plein de boules dedans (pour l'instant fixes)
# étudier mathématiquement le bail, gaz de van der waals etc + exposant de lyapunov
# on commence en dimension 2 (dimension n etc..)

from math import acos, asin, inf,cos,sin,pi
from random import random


h,L=2,2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


boules = [(np.array([[1+random()],[1+random()]]),0.3*random()) for _ in range(4)]

état=(0,0)
def signe(x):
    if x<0:
        return -1
    return 1

def norm(x):
    return np.linalg.norm(x)

def discriminant(etat,i):
    a,b,c=norm(etat[1])**2,np.dot(np.transpose(-1*boules[i][0]+etat[0]),etat[1])[0][0],-boules[i][1]+ norm(boules[i][0])**2+norm(etat[0])**2-2*np.dot(np.transpose(boules[i][0]), etat[0])[0][0]
    delta=b*b-4*a*c
    return a,b,c,delta
def collision(i,etat): # regard s'il va y avoir une collision avec la boule i dans
    a,b,c,delta= discriminant(etat,i)
    if delta<=0:
        return inf
    return  racine_plusproche_0(a,b,c,delta)
    
    
def mat_rot(t):
    return np.array([[cos(t),-sin(t)],[sin(t),cos(t)]])

def racine_plusproche_0(a,b,c,delta):
    x,y=(-b+delta)/(2*a),(-b-delta)/(2*a)
    if abs(x)<abs(y):
        return x
    return y


def indice_min(L):
    i=0
    m=inf
    for j in range(len(L)):
        if L[j]<m:
            i=j
            m=L[i]
    return i    



def angle(i,impact,etat):
    c=boules[i][0]
    m=impact-c
    return asin(np.linalg.det(np.array([[m[0][0],-etat[1][0][0]],[m[1][0],-etat[1][1][0]]]))/(norm(etat[1])*norm(m)))
    
def limite(etat):
    x,y=etat[0][0][0],etat[0][1][0]
    vx,vy=etat[1][0][0],etat[1][1][0]
    print(vx,vy)
    test1 = np.dot(np.transpose(etat[1]),np.array([[1],[0]]))[0][0] >0
    test2=np.dot(np.transpose(etat[1]),np.array([[0],[1]]))[0][0]>0
    if test1:
        if test2:
            tmur=abs((L-x)/vx)
            thaut=abs((h-y)/vy)
            if tmur>thaut:
                etat[0]=etat[0]+thaut*etat[1]
                etat[1][1]=-vy
            else:
                etat[0]=etat[0]+tmur*etat[1]
                etat[1][1]=-vx  
        else:
            tmur=abs((L-x)/vx)
            thaut=abs(y/vy)
            if tmur>thaut:
                etat[0]=etat[0]+thaut*etat[1]
                etat[1][1]=-vy
            else:
                etat[0]=etat[0]+tmur*etat[1]
                etat[1][1]=-vx
        
    else:
        if test2:
            tmur=abs(x/vx)
            thaut=abs((h-y)/vy)
            if tmur>thaut:
                etat[0]=etat[0]+thaut*etat[1]
                etat[1][1]=-vy
            else:
                etat[0]=etat[0]+tmur*etat[1]
                etat[1][1]=-vx
        else:
            tmur=abs(x/vx)
            thaut=abs(y/vy)
            if tmur>thaut:
                etat[0]=etat[0]+thaut*etat[1]
                etat[1][1]=-vy
            else:
                etat[0]=etat[0]+tmur*etat[1]
                etat[1][1]=-vx
        
def évolue(etat):
    L=[collision(i,etat) for i in range(len(boules))]
    i= indice_min(L)
    if L[i]==inf : # cas aucune collision 
        limite(etat)
    else:
        impact=etat[0]+L[i]*etat[1]
        alpha = angle(i,impact,etat)
        etat[1]=mat_rot(-((pi-2*alpha)*signe((impact-boules[i][0])[0])))@etat[1]
        etat[0]=impact


resolution = 100 # 100 point par unité de longueur

def parcours(etat_init,n):
    L=[etat_init]
    for _ in range(n):
        évolue(etat_init)
        L.append(etat_init)
    return L


def mise_en_forme(x):
    M=[x[0]]
    for i in range(len(x)-1):
        a=M[-1]
        print(x[i+1])
        M=M+[(1-k/resolution)*a + x[i+1]*k/resolution for k in range(resolution+1)]
    return


M=parcours(np.array([np.array([[0],[0]]),np.array([[1],[0.5]])]),20)


M=mise_en_forme(M)


fig,ax= plt.subplots() # initialise la figure
ax.grid()
line, = plt.plot([],[]) 


for b in boules:
    X,Y=[b[0][0][0]+b[1]*cos(2*pi*k/resolution) for k in range(resolution+1)],[b[0][1][0]+b[1]*sin(2*pi*k/resolution) for k in range(resolution+1)]
    plt.plot(X,Y)

def init():
    line.set_data([],[])
    return line,


def animate(i): # fonction majeure
    line.set_data(M[10*i:20*i][0],M[10*i:20*i][1])
    return line,
 
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=10000, blit=True, interval=20, repeat=True)

plt.show()





        
    