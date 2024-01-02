from math import sin,cos, tan, pi
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

R,l0,k,m1,m2=0.5,0.5,10,1,0.2

# version numpy



# version rk4 (avec n paramètres) pour tester si c'est meilleur


def rk4vect(F,arrayinit,h,n):  # F la fonction big vecteur -> big vecteur
    L=np.ndarray((n,8),dtype=float)
    L[0,:]=arrayinit
    z=arrayinit
    for k in range(1,n):
        k1=F(z)
        k2=F(z+h/2*k1)
        k3=F(z+h/2*k2)
        k4=F(z+h*k3)
        z+=h/6*(k1+2*k2+2*k3+k4)
        L[k,:]=z
    return L

def F(vect):
    t,tp,r,rp,theta,thetap,phi,phip= vect[0],vect[1],vect[2],vect[3],vect[4],vect[5],vect[6],vect[7]
    return np.array([tp,-k*(r-l0)*sin(theta)*(-sin(t)*cos(phi)+cos(t)*sin(phi)),rp, -k*(r-l0)+r*phip*phip*sin(theta)*sin(theta),thetap,-2*rp/r+phip*phip*sin(theta)*cos(theta) ,phip,-2*phip*rp/r-2*phip*thetap/tan(theta)],dtype=float)


def renormalisation(T,mu): # x1,y1,x2,y2,z2
    a,b=np.shape(T)
    L=np.ndarray((a//mu,6),dtype=float)
    for k in range(0,b,mu):
        L[k,0]=R*cos(T[k,0])
        L[k,1]=R*sin(T[k,0])
        L[k,2]=0
        L[k,3]=L[k,0]+T[k,2]*sin(T[k,4])*cos(T[k,5])
        L[k,4]=L[k,1]+T[k,2]*sin(T[k,4])*sin(T[k,5])
        L[k,5]=T[k,2]*cos(T[k,4])
    return L
        


def experience(vectinit,h,N,facteurderenormalisation):# N le nombre d'itérations
    T=rk4vect(F,vectinit,h,N)
    # renormalisation en coordonnées cartésiènnes
    H=renormalisation(T,facteurderenormalisation)

    def func(num, dataSet, line):
        line.set_data(dataSet[:num,0:2])    
        line.set_3d_properties(dataSet[:num,2])    
        return line

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    # Setting the axes properties
    ax.set_xlim3d([-1, 1.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([-1, 1.0])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-1, 1.0])
    ax.set_zlabel('Z')

    ax.set_title('Pendule ressort sur base circulaire fixe')
 
    dataSet1 = np.copy(H[:,0:3])
    dataSet2 = np.copy(H[:,3:6])
    numDataPoints = len(H[:,0])

 
    # NOTE: Can't pass empty arrays into 3d version of plot()
    line1 = plt.plot(dataSet1[:,0], dataSet1[:,1], dataSet1[:,2], lw=2, c='black')[0]
    line2 = plt.plot(dataSet2[:,0], dataSet2[:,1], dataSet2[:,2], lw=2, c='black')[0]
 
    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, func, frames=numDataPoints, fargs=(dataSet2,line2), interval=50, blit=False)
    #line_ani.save(r'AnimationNew.mp4')
 
 
    plt.show()
    
    
    
    



experience(np.array([1,0,0.5,0,pi/2,0,0,0],dtype=float),0.001,100000,5)







# idée
# faire une étude exhaustive des points de stabilité, des périodes, évaluer des écarts etc ..
# bref voir pour quelle masse et quel paramètre ça va être stable ou faire des trucs de dingues
# faire des fonctions de mesure de stabilité
