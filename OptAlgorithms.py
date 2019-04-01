import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from NesterovModule import Nesterov
from PolyakModule import Polyak 


from mpl_toolkits.mplot3d import Axes3D



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
                    Dimension Parameter
'''

N = 2 



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
                    Objective Functions Selection
'''

indexFct = input('Choose the convex / non-convex function : ')
indexFct = int(indexFct)
print('\n\n')

def FctCases( i ) :
        switcher = {
                0 : 'Convex 1D',
                1 : 'Booth Function',
                2 : 'McCormik',
                3 : 'Exponential Function',
                4 : 'Schwefel 2.23',
                5 : 'Rastrigin', 
                6 : 'Sum Squares Function'
             }
        print( switcher[i], '\n' )
        return switcher[i]
            

def FctDimension( i ) :  
    if i == 0 :
        return 1
    else :
        return 2

titleFct = FctCases( indexFct ) 
N = FctDimension( indexFct )



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
                    Objective Functions
'''

def E(x) :
    if indexFct == 0 :
        return x[0] ** 4
    
    elif indexFct == 1 :
        return ( x[0] + 2 * x[1] - 7 ) ** 2 + ( 2 * x[0] + x[1] - 5 ) ** 2
    
    elif indexFct == 2 :
        return np.sin( x[0] + x[1] ) + ( x[0] - x[1] ) ** 2 - 1.5 * x[0] + 2.5 * x[1] + 1
    
    elif indexFct == 3 : 
        return - np.exp( - 0.5 * ( x[0] ** 2 + x[1] ** 2 ) )
    
    elif indexFct == 4 :
        return x[0] ** 10 + x[1] ** 10   
    
    elif indexFct == 5 :
        return 10 * N + ( x[0] ** 2 - 10 * np.cos( 2 * np.pi * x[0] ) ) + ( x[1] ** 2 - 10 * np.cos( 2 * np.pi * x[1] ) )    
    
    elif indexFct == 6 : 
        return x[0] ** 2 + 2 * x[1] ** 2    


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
                    Gradient of the Objective Functions
'''

def gradE(x) :
    if indexFct == 0 :
        return 4 * x[0] ** 3
    
    elif indexFct == 1 :
        return [ 10 * x[0] + 8 * x[1] - 34, 8 * x[0] + 10 * x[1] - 38 ]
    
    elif indexFct == 2 :
        return [ np.cos( x[0] + x[1] ) + 2 * x[0] - 2 * x[1] - 1.5 , np.cos( x[0]+x[1] ) - 2 * x[0] + 2 * x[1] + 2.5 ]
    
    elif indexFct == 3 : 
        return [ x[0] * np.exp( -0.5 * x[0] ** 2 - 0.5 * x[1] ** 2 ) , x[1] * np.exp( -0.5 * x[0] ** 2 - 0.5 * x[1] ** 2 ) ]
    
    elif indexFct == 4 :
        return [ 10 * x[0] ** 9 , 10 * x[1] ** 9 ]  
    
    elif indexFct == 5 :
        return [ 2 * x[0] + 20 * np.pi * np.sin( 2 * np.pi * x[0] ) , 2 * x[1] + 20 * np.pi * np.sin( 2 * np.pi * x[1] ) ]    
    
    elif indexFct == 6 : 
        return [ 2 * x[0] , 4 * x[1] ]
    
    
    
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
                    Plot Functions for N <= 2
'''

def plotE(x, y) :
    if indexFct == 0 :
        return x ** 4
    
    elif indexFct == 1 :
        return ( x + 2 * y - 7 ) ** 2 + ( 2 * x + y - 5 ) ** 2
    
    elif indexFct == 2 :
        return np.sin( x + y ) + ( x - y ) ** 2 - 1.5 * x + 2.5 * y + 1
    
    elif indexFct == 3 : 
        return - np.exp( - 0.5 * ( x ** 2 + y ** 2 ) )
    
    elif indexFct == 4 :
        return x ** 10 + y ** 10

    elif indexFct == 5 :
        return 10 * N + ( x ** 2 - 10 * np.cos( 2 * np.pi * x ) ) + ( y ** 2 - 10 * np.cos( 2 * np.pi * y ) )  
    
    elif indexFct == 6 : 
        return x ** 2 + 2 * y ** 2
    
    
    
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
                    Function Plots
'''



if N == 1 :
    fig = plt.figure()
    x = np.linspace(-5, 5, 100)
    lstOx = [x]
    lstOy = E(lstOx)
    plt.plot(x, lstOy)
    plt.xlabel(r'$ x $', fontsize = 12)
    plt.ylabel(r'$ E(x) $', fontsize = 12)
    plt.title('Objective Function')
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    
elif N == 2 :
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, x)
    Z = plotE(X, Y)
    ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1,cmap = 'viridis', edgecolor = 'none')
    ax.set_xlabel(r'$ x $', fontsize = 12)
    ax.set_ylabel(r'$ y $', fontsize = 12)
    ax.set_zlabel(r'$ E(x,y) $', fontsize = 12)
    ax.set_title(titleFct);
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()    
    
    fig = plt.figure()
    if indexFct == 1 :
        x = np.linspace(0, 2, 1000)
        y = np.linspace(2, 4, 1000)
    elif indexFct == 2 :
        x = np.linspace(-1.6, 1.6, 1000)
        y = np.linspace(-2.6, 2.6, 1000)
    elif indexFct == 3 :    
        x = np.linspace(-1, 1, 1000)
        y = np.linspace(-1, 1, 1000)
    elif indexFct == 4 :
        x = np.linspace(-2, 2, 1000)
        y = np.linspace(-2, 2, 1000)
    elif indexFct == 5 :
        x = np.linspace(-4, 4, 1000)
        y = np.linspace(-4, 4, 1000)
    elif indexFct == 6 :
        x = np.linspace(-4, 4, 1000)
        y = np.linspace(-4, 4, 1000)    
        
    X, Y = np.meshgrid(x, y)
    Z = plotE(X, Y)
    plt.contourf(X, Y, Z, 20, cmap='RdGy')
    plt.colorbar();
    plt.title(titleFct + ' Contour Plot')
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    
else :
    print('Dimension is much higher than 2. Unable to make 3d plot ... \n \n')    




# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
                    Initial Values
'''

initialValue = 2

u0 = initialValue * np.ones((1, N))
u0 = u0[0]

v0 = np.zeros((1, N))
v0 = v0[0]



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
                    Parameters
'''


eps = 10 ** ( -5 )
gamma = 0.9

stepSize_Nesterov = 0.1
stepSize_Polyak = 0.1



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
                    Minimum Points
'''

if indexFct == 0 :
    minPoint = 0 

elif indexFct == 1 :
    minPoint = [ 1 , 3 ]    
    
elif indexFct == 2 :
    minPoint = [ -0.547 , -1.547 ]    

elif indexFct == 3 :
    minPoint = [ 0 , 0 ]

elif indexFct == 4 :
    minPoint = [ 0 , 0 ]

elif indexFct == 6 :
    minPoint = [ 0 , 0]



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
                    Algorithms
'''


itN, uN, vN = Nesterov(u0, v0, stepSize_Nesterov, eps, E, gradE, N)
itP, uP, vP = Polyak(u0, v0, stepSize_Polyak, eps, E, gradE, N, gamma)


5

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
                    Energy Decay, Iterations, Energy Dissipation
'''


itN = np.array(itN)
timeN = np.linspace( 1, len(itN), len(itN) ) * stepSize_Nesterov
uN = np.array(uN)
vN = np.array(vN)

if N == 1 :
    matrix_uN = np.array([uN,[]])
    EN = E( matrix_uN )
    ChangeE_N = EN[1:] - EN[0:-1]
else :
    matrix_uN = [uN[:,0] , uN[:,1] ]
    EN = E( matrix_uN )
    ChangeE_N = EN[1:] - EN[0:-1]
    



itP = np.array(itP)
timeP = np.linspace( 1, len(itP), len(itP) ) * stepSize_Polyak
uP = np.array(uP)
vP = np.array(vP)

if N == 1 :
    matrix_uP = np.array([uP,[]])
    EP = E( matrix_uP )
    ChangeE_P = EP[1:] - EP[0:-1]
else :
    matrix_uP = [uP[:,0] , uP[:,1] ]
    EP = E( matrix_uP )
    ChangeE_P = EP[1:] - EP[0:-1]     


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
                    Plots
'''
  

my_colors = ['red', 'yellow']
my_markers = ['d', 'o']




if N == 1 :
    fig = plt.figure()
    plt.plot(itN, uN, color = my_colors[0], marker = my_markers[0], markersize = 8, label = 'Nesterov')
    plt.plot(itP, uP, color = my_colors[1], marker = my_markers[1], markersize = 8, label = 'Polyak')
    plt.xlabel(r' $ Iterations $ ', fontsize = 12)
    plt.ylabel(r' $ u_{n} $ ', fontsize = 12)
    plt.title(titleFct + ' Iteration Decay')
    plt.legend()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()

    fig = plt.figure()
    plt.plot(itN, EN, color = my_colors[0], marker = my_markers[0], markersize = 8, label = 'Nesterov')
    plt.plot(itP, EP, color = my_colors[1], marker = my_markers[1], markersize = 8, label = 'Polyak')
    plt.xlabel(r' $ Iterations $ ', fontsize = 12)
    plt.ylabel(r' $ E(u_{n}) $ ', fontsize = 12)
    plt.title(titleFct + ' Energy Decay')
    plt.legend()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    
    fig = plt.figure()
    plt.plot(timeN[1:], ChangeE_N, color = my_colors[0], marker = my_markers[0], markersize = 8, label = 'Nesterov')
    plt.plot(timeP[1:], ChangeE_P, color = my_colors[1], marker = my_markers[1], markersize = 8, label = 'Polyak')
    plt.xlabel(r' $ t = n h $ ', fontsize = 12)
    plt.ylabel(r' $ E(u_{n+1}) - E(u_{n}) $ ', fontsize = 12)
    plt.title(titleFct + 'Energy Dissipation')
    plt.legend()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    
    fig = plt.figure()
    x = np.linspace(-5, 5, 100)
    lstOx = [x]
    lstOy = E(lstOx)
    plt.plot(x, lstOy)
    plt.plot(uN, EN, color = my_colors[0], marker = my_markers[0], markersize = 8, label = 'Nesterov')
    plt.plot(uP, EP, color = my_colors[1], marker = my_markers[1], markersize = 8, label = 'Polyak')
    plt.xlabel(r' $ u_{n+1} $ ', fontsize = 12)
    plt.ylabel(r' $ E(u_{n+1}) - E(u_{n}) $ ', fontsize = 12)
    plt.title(titleFct + ' Function and Values')
    plt.legend()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    
    fig = plt.figure()
    nrItems = 2
    ax = plt.gca()
    ax.tick_params(axis = 'x', colors = 'blue')
    ax.tick_params(axis = 'y', colors = 'green')
    index = np.arange(nrItems)
    width = 0.8
    values = ( len(itN), len(itP) )
    new_values = np.sort(values)
    names = ('Nesterov', 'Polyak')
    new_my_colors = list((x for _,x in sorted(zip(values, my_colors))))
    new_names = list((x for _,x in sorted(zip(values, names))))
    plt.bar(index, new_values, width, color = new_my_colors, edgecolor = 'black')
    plt.xticks(index, new_names)
    plt.title(r'Iterations for $ \varepsilon = $' + str(eps))
    
    
else :
    fig = plt.figure()
    plt.plot(itN, uN[:,0], color = my_colors[0], marker = my_markers[0], markersize = 8, label = 'Nesterov')
    plt.plot(itP, uP[:,0], color = my_colors[1], marker = my_markers[1], markersize = 8, label = 'Polyak')
    plt.xlabel(r' $ Iterations $ ', fontsize = 12)
    plt.ylabel(r' $ u_{n,1} $ ', fontsize = 12)
    plt.title(titleFct + ' Iteration Decay')
    plt.legend()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()

    fig = plt.figure()
    plt.plot(itN, uN[:,1], color = my_colors[0], marker = my_markers[0], markersize = 8, label = 'Nesterov')
    plt.plot(itP, uP[:,1], color = my_colors[1], marker = my_markers[1], markersize = 8, label = 'Polyak')
    plt.xlabel(r' $ Iterations $ ', fontsize = 12)
    plt.ylabel(r' $ u_{n,2} $ ', fontsize = 12)
    plt.title(titleFct + ': Iteration Decay')
    plt.legend()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    
    fig = plt.figure()
    plt.plot(itN, EN, color = my_colors[0], marker = my_markers[0], markersize = 8, label = 'Nesterov')
    plt.plot(itP, EP, color = my_colors[1], marker = my_markers[1], markersize = 8, label = 'Polyak')
    plt.xlabel(r' $ Iterations $ ', fontsize = 12)
    plt.ylabel(r' $ E(u_{n}) $ ', fontsize = 12)
    plt.title(titleFct + ' Energy Decay')
    plt.legend()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    
    fig = plt.figure()
    plt.plot(timeN[1:], ChangeE_N, color = my_colors[0], marker = my_markers[0], markersize = 8, label = 'Nesterov')
    plt.plot(timeP[1:], ChangeE_P, color = my_colors[1], marker = my_markers[1], markersize = 8, label = 'Polyak')
    plt.xlabel(r' $ t = n h $ ', fontsize = 12)
    plt.ylabel(r' $ E(u_{n+1}) - E(u_{n}) $ ', fontsize = 12)
    plt.title(titleFct + ' Energy Dissipation')
    plt.legend()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    
    fig = plt.figure()
    if indexFct == 1 :
        x = np.linspace(0, 2, 1000)
        y = np.linspace(2, 4, 1000)
    elif indexFct == 2 :
        x = np.linspace(-1.6, 1.6, 1000)
        y = np.linspace(-2.6, 2.6, 1000)
    elif indexFct == 3 :    
        x = np.linspace(-1, 1, 1000)
        y = np.linspace(-1, 1, 1000)
    elif indexFct == 4 :
        x = np.linspace(-2, 2, 1000)
        y = np.linspace(-2, 2, 1000)
    elif indexFct == 5 :
        x = np.linspace(-4, 4, 1000)
        y = np.linspace(-4, 4, 1000)
    elif indexFct == 6 :
        x = np.linspace(-4, 4, 1000)
        y = np.linspace(-4, 4, 1000)    
        
    X, Y = np.meshgrid(x, y)
    Z = plotE(X, Y)
    plt.contourf(X, Y, Z, 20, cmap='RdGy')
    plt.colorbar();
    plt.plot(uN[:,0], uN[:,1], color = my_colors[0], marker = my_markers[0], markersize = 8, label = 'Nesterov')
    plt.plot(uP[:,0], uP[:,1], color = my_colors[1], marker = my_markers[1], markersize = 8, label = 'Polyak')
    plt.xlabel(r' $ u_{n,2} $ ', fontsize = 12)
    plt.ylabel(r' $ u_{n,2} $ ', fontsize = 12)
    plt.title(titleFct + ' Phase Portrait')
    plt.legend()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    
    
    fig = plt.figure()
    nrItems = 2
    ax = plt.gca()
    ax.tick_params(axis = 'x', colors = 'blue')
    ax.tick_params(axis = 'y', colors = 'green')
    index = np.arange(nrItems)
    width = 0.8
    values = ( len(itN), len(itP) )
    new_values = np.sort(values)
    names = ('Nesterov', 'Polyak')
    new_my_colors = list((x for _,x in sorted(zip(values, my_colors))))
    new_names = list((x for _,x in sorted(zip(values, names))))
    plt.bar(index, new_values, width, color = new_my_colors, edgecolor = 'black')
    plt.xticks(index, new_names)
    plt.title(r'Iterations for $ \varepsilon = $' + str(eps))