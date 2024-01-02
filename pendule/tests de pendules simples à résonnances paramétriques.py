from math import cos,sin,tan,pi


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


def afficher6_temp(F,c,a,h,n): # portrait de phase simple
    import matplotlib.pyplot as plt
    import numpy as np
    ax=plt.subplots()[1]
    ax.set(xlabel='x', ylabel='y')
    ax.grid()
    b1,b2=rk4vect_temp(F,c[0],a,h,n)
    plt.plot(b1,b2,color='black')
    plt.show()


# paramètres 
# a grand permet de rendre moins sensible aux perturbations sinusoidales
a=0.3 # position d'équilibre
q=9 # amplitude
T=1 # période
frottements=0


# on a testé un truc avec tangente, ça n'a pas trop fonctionné
def g(t):
    return a-2*q*cos(cos(2*pi*t/T))

def f(x,y,t): # la constante sur le t² est un paramètre de l'équation
    return y,-x*g(t)-frottements*y


afficher5_temp(f,(1,1),(1,-1),0.000001,30000000)


afficher6_temp(f,(1,1),(1,-1),0.000001,30000000)

# système assez sensible aussi, faire attentoin à garder un pas assez petit sinon ça diverge grossièrement