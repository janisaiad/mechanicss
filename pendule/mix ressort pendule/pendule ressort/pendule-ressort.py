from math import pi
# brouillon rose


#constantifier tout ça


def rk4vect_dim4(F,vectinit,h,n):
    L1,L2,L3,L4,z=[vectinit[0]],[vectinit[1]],[vectinit[2]],[vectinit[3]],vectinit
    for _ in range(1,n):
        p,q,r,s=z[0],z[1],z[2],z[3]
        k1,kk1,kkk1,kkkk1=F(p,q,r,s)
        k2,kk2,kkk2,kkkk2=F(p+h/2*k1,q+h/2*kk1,r+h/2*kkk1,s+h/2*kkkk1)
        k3,kk3,kkk3,kkkk3=F(p+h/2*k2,q+h/2*kk2,r+h/2*kkk2,s+h/2*kkkk2)
        k4,kk4,kkk4,kkkk4=F(p+h*k3,q+h*kk3,r+h*kkk3,s+h*kkkk3)
        z=p+h/6*(k1+2*k2+2*k3+k4),q+h/6*(kk1+2*kk2+2*kk3+kk4),r+h/6*(kkk1+2*kkk2+2*kkk3+kkk4),s+h/6*(kkkk1+2*kkkk2+2*kkkk3+kkkk4)
        L1.append(z[0])
        L2.append(z[1])
        L3.append(z[2])
        L4.append(z[3])
    return L1,L2,L3,L4

def afficher_positionsvitesse(F,c,a,h,n): # graphe des fonctions positions et vitesse
    import matplotlib.pyplot as plt
    import numpy as np
    from math import cos,sin
    ax=plt.subplots()[1]
    ax.set(xlabel='x', ylabel='y')
    ax.grid()
    p=np.linspace(c[0],c[0]+n*h,n) # l0,lpoint0,theta0,thetapoint0
    q=np.linspace(c[1],c[1]+n*h,n)
    r=np.linspace(c[2],c[2]+n*h,n)
    s=np.linspace(c[3],c[3]+n*h,n)
    l,lpoint,theta,thetapoint=rk4vect_dim4(F,a,h,n)
    plt.plot(p,l,color='green')
    plt.plot(q,lpoint,color='blue')
    plt.plot(r,theta,color='black')
    plt.plot(s,thetapoint,color='grey')
    plt.show()
    X,Y=[],[]
    for i in range(len(l)):
        X+=[-l[i]*sin(theta[i])]
        Y+=[-l[i]*cos(theta[i])]
    return X,Y

def pendule_ressort(x,y,u,v): # revoir la formule pour reprendr eles valeurs caractéristiques des chiffres, 0.2 est sûrement m,1gauche est l0, 5 est k
    from math import cos,sin
    return y,5/0.2*(1-x)+9.81*cos(u)+x*v*v,v,(-9.81*sin(u)-2*y*v)/(x*1)

# afficher_positionsvitesse(pendule_ressort,(0,0,0,0),(1,0,pi/4,0),0.0001,50000)




def portrait_de_phase(F,a,h,n): # portrait de phase simple
    import matplotlib.pyplot as plt
    import numpy as np
    ax=plt.subplots()[1]
    ax.grid()
    l,lpoint,theta,thetapoint=rk4vect_dim4(F,a,h,n)
    plt.plot(l,lpoint)
    plt.show()
    ax=plt.subplots()[1]
    ax.grid()
    plt.plot(theta,thetapoint)
    plt.show()

# portrait_de_phase(pendule_ressort,(1,0,pi/3,0),0.0001,500000)





















# temporaire = ((0,0,0,0),(1,-4,pi/1.5,-7),0.0001,100000)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#  X,Y=afficher_positionsvitesse(pendule_ressort,(0,0,0,0),(0.3,1,pi/1.5,0.964327773640),0.0001,100000) donne des choses très cool, moyen de calculer la constante directement ?
# revoir equadiff demange sur un programme de resolution en rk2 ou un truc comme ça où ça pète comme du popcorn
X,Y=afficher_positionsvitesse(pendule_ressort,(0,0,0,0),(1,0,pi/1.5,1),0.0002,400000) #l0, lpoint0, theta0, theta point0
# pour 0.96 ça refait le tour en arrière

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
    line.set_data(X[50*i:100*i]+[0],Y[50*i:100*i]+[0])
    return line,
 
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=2000, blit=True, interval=10, repeat=False)

plt.show()