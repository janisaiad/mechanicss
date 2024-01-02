import numpy as np
from math import pi,cos,sin,exp,atan
import matplotlib.pyplot as plt
import matplotlib.animation as animation


m,L,g,tau,frott,h=1,1.4,9.81,4,0.5,0.7
J,l=(h*h*h/3+L*L*L/4+L*h*h)/(L+h),(h/2+L)/(1+L/h)
psi=atan(L/(2*h))
w0=10
R=((L/2)*(L/2)+h*h)**0.5
D0=2
# schéma simple pour reprendre les bases


# tout écrire en vectoriel avec numpy pour la 2eme fois et on étudiera en 3d le trucs

def rk4vect_dim4(F,vectinit,h,n):
    L,z=np.zeros((n,3)),vectinit
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
    plt.plot(L[:,0],L[:,2]/4,color='blue')
    plt.show()
    return L[:,0],L[:,1] # on renvoie theta



def matrix(t):
    return
    
def omega(t):
    return w0*(1-exp(-t/tau))+w0

def omegaprime(t):
    return -w0*1/tau*exp(-t/tau)

# paramétrer en fonction du noombre de rotations (chose à généraliser pour toutes les études)
def evolue(z):
    om=omega(z[0])
    return np.array([1,z[2],-m*g*l*sin(z[1])+l*omegaprime(z[0])*cos(z[1])+om*om*l*cos(z[1])-frott*z[2]])









T,theta=afficher_positionsvitesse(evolue,np.array([0,pi-1,10]),0.001,20000)


fig,ax= plt.subplots() # initialise la figure
ax.grid()
line, = plt.plot([],[]) 
plt.xlim(-6,6)
plt.ylim(-6,6)
"""

plt.plot([-10,10],[0,0],c='black') # sol


# fonction à définir quand blit=True

defn=1000 # nombre de points pour la création du cercle

plt.plot([R*cos(2*pi*k/defn+pi/2) for k in range(defn+1)],[R*sin(2*pi*k/defn+pi/2) for k in range(defn +1)])


# crée l'arrière de l'animation qui sera présent sur chaque image
def init():
    line.set_data([],[])
    return line,

def animate(i):
    line.set_data([R*cos(theta[10*i]-psi-pi/2),R*cos(theta[10*i]+psi-pi/2)],[R*sin(theta[10*i]-psi-pi/2),R*sin(theta[10*i]+psi-pi/2)])
    return line,
 
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=1000, blit=True, interval=10, repeat=False)
plt.show()

"""
# maintenant on fabrique les projections

def norm(x):
    return sum(j[0]**2 for j in x )**0.5
x,y,z=1,1,0.1
vect = np.zeros((1,3))
vect[0][0]=x
vect[0][1]=y
vect[0][2]=z
vectbis= np.array([y,-x,0])
M=np.identity(3)-np.matmul(np.transpose(vect),vect)*(norm(vect))**(-2)
print(M)
def projette(z):
    u=np.matmul(M,z)
    return (-y*u[0][0]+x*u[1][0])/(norm(vect)),u[2][0]


defn=150 # nombre de points pour la création du cercle

h=[projette(np.array([[0],[0],[(D0-R)*k/defn]])) for k in range(defn)]

h2=[projette(np.array([[0],[(D0-R)*k/defn],[0]])) for k in range(defn)]

h3=[projette(np.array([[(D0-R)*k/defn],[0],[0]])) for k in range(defn)]
plt.plot([k[0] for k in h],[k[1] for k in h])
plt.plot([k[0] for k in h2],[k[1] for k in h2])
plt.plot([k[0] for k in h3],[k[1] for k in h3])
# crée l'arrière de l'animation qui sera présent sur chaque image
def init():
    line.set_data([],[])
    return line,



def animate(i):
    liste = [np.array([[R*sin(2*pi*k/defn+theta[10*i]-psi)*cos(omega(T[10*i])*T[10*i])],[R*sin(2*pi*k/defn+theta[10*i]-psi)*sin(omega(T[10*i])*T[10*i])],[D0-R*cos(2*pi*k/defn+theta[10*i]-psi)]]) for k in range(defn +1)]
    a=[projette(u) for u in liste]

    new2=projette(np.array([[R*sin(theta[10*i]+psi)*cos(omega(T[10*i])*T[10*i])],[R*sin(theta[10*i]+psi)*sin(omega(T[10*i])*T[10*i])],[D0-R*cos(theta[10*i]+psi)]]))
    line.set_data([k[0] for k in a]+[new2[0]],[k[1] for k in a]+[new2[1]])
    return line,
 
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=1000, blit=True, interval=10, repeat=False)
plt.show()



