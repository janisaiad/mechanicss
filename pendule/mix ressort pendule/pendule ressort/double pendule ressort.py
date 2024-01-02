from math import pi,cos,sin,atan
# brouillon rose 


# paramètres
x,y = 1 ,0
L=(x*x+y*y)**0.5
gamma=atan(y/x)
m,g,k1,k2,l01,l02 =0.8,9.81,5,20,0.5*2**0.5,0.5*2**0.5


# frottements
frott=0.01

# schéma d'intégration
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
        X+=[l[i]*sin(theta[i])]
        Y+=[-l[i]*cos(theta[i])]
    return X,Y




# fonction principale du système
def double_pendule_ressort(r1,r1point,theta1,theta1point): # on note prov = cos(theta1+theta2), prov2=sin(theta1+theta2)
    # on utilise la loi des sinus très facile à prouver
    v=(r1point**2+r1*theta1point**2)**0.5
    r2=(r1*r1+L*L+2*r1*L*sin(gamma-theta1))**0.5
    prov=(1/(2*r1*r2))*(-L*L+r1*r1+r2*r2)
    prov2=(L/r2)*cos(gamma-theta1)
    return r1point,g*cos(theta1)+r1*theta1point*theta1point-(k1/m)*(r1-l01)-(k2/m)*(r2-l02)*prov-(frott/m)*r1point , theta1point,1/r1*(-g*sin(theta1)+(k2/m)*(r2-l02)*prov2-2*theta1point*r1point-(frott/m)*r1*theta1point)

# afficher_positionsvitesse(pendule_ressort,(0,0,0,0),(1,0,pi/4,0),0.0001,50000)



# traçage du portrait de phase
def portrait_de_phase(F,a,h,n): # portrait de phase simple
    import matplotlib.pyplot as plt
    ax=plt.subplots()[1]
    ax.grid()
    l,lpoint,theta,thetapoint=rk4vect_dim4(F,a,h,n)
    plt.plot(l,lpoint,color="black")
    plt.show()
    ax=plt.subplots()[1]
    ax.grid()
    plt.plot(theta,thetapoint,color="black")
    plt.show()

# portrait_de_phase(double_pendule_ressort,(2,0,-pi/1.3,0),0.00001,1000000)























# animation 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

X,Y=afficher_positionsvitesse(double_pendule_ressort,(0,0,0,0),(0.2*2**0.5,3,-3*pi/1.6,6),0.0001,100000) #l0, lpoint0, theta0, theta point0

fig,ax= plt.subplots() # initialise la figure
ax.grid()
line, = plt.plot([],[]) 
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.plot([0,x],[0,y],color="black")
# fonction à définir quand blit=True, toujours pas compris cette remarque

# crée l'arrière de l'animation qui sera présent sur chaque image
def init():
    line.set_data([],[])
    return line,

def animate(i): # fonction majeure
    if i == 0:
        line.set_data(X[0:1]+[x],Y[0:1]+[y])
    else:

        line.set_data(X[400*i:500*i]+[0]+[X[500*i-1]]+[x],Y[400*i:500*i]+[0]+[Y[500*i-1]]+[y])
    return line,
 
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=1000, blit=True, interval=10, repeat=True)

plt.show()

# approfondissements
# faire dépendre y et x du temps, recopier tout le code dans la partie ci-dessous et étudier ce phénomène et comment le mettre en oeuvre
# approfondir avec les ressources trouvées sur des sites sur google, avec des simulateurs déjà fait, 
# on peut approfondir avec des pendules couplés par un ressort entre les deux
# inversement, deux pendules ressorts décalés qui sont couplés par une tige, ie la distance entre les deux extré^mités est constante
# des pendules simples et doubles, à entrainement circulaire aussi, faire des diagrammes ou des trucs en volumes pour étudier les paramètres qui rendent le tout chaotique etc.. en étudiant l'amplitude en angle des oscillations voir https://fr.wikipedia.org/wiki/Pendule_double#Pendule_%C3%A0_entra%C3%AEnement_circulaire_uniforme
# faire une catégorie de marques pages et de fichier ici DEDIE AUX PENDULES EN TOUT GENRE