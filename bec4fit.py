import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import curve_fit
from scipy.signal import convolve2d
from scipy.signal import medfilt2d

def gaussian( x, *p):
    """
    1D Gaussian

    p: Amplitude, x0, width, offset

    """
    return p[0] * np.exp( -(x - p[1])**2 / (2 * p[2]**2) ) + p[3]

def gaussian_sloped(x,*p):
    """
    1D Gaussian with linear slope

    p: Amplitude, x0, width, offset, slope

    """
    return p[0] * np.exp( -(x - p[1])**2 / (2 * p[2]**2) ) + p[4]*x + p[3]

def gaussian2D(x,y,x0,y0,sx,sy,A,B):
    """
    2D Gaussian

    x,y: 2d meshgrid
    x0,y0: centers
    sx,sy: widths
    A: amplitude
    B: offset

    """
    return A * np.exp(-0.5*((x-x0)/sx)**2 - 0.5*((y-y0)/sy)**2) + B

def background2D(x,y,a,b,c):
    """
    2D background surface

    a: x-slope
    b: y-slope
    c: constant offset

    """
    return a*x + b*y + c

def _background2D(M,*args):
    """
    function passed to curve_fit to use background2D
    """
    x,y = M
    return background2D(x,y,*args)

def _gaussian2D(M, *args):
    """
    function passed to curve_fit to use gaussian2D
    """
    x, y = M
    return gaussian2D(x, y, *args)

def gaussian2Drot(x,y,x0,y0,sx,sy,A,B,theta):
    """
    rotated 2D Gaussian

    x0,y0: centers
    sx,sy: widths
    A: amplitude
    B: offset
    theta: rotation angle

    """
    return A*np.exp((1/(2*sx*sy)**2)*(-(sx**2 + sy**2)*((x-x0)**2 + (y-y0)**2) 
                                      + (sx-sy)*(sx+sy)*((x - x0 + y - y0)*(x - x0 - y + y0)*np.cos(2*theta)
                                      + 2*(x - x0)*(y - y0)*np.sin(2*theta)))) + B

def _gaussian2Drot(M, *args):
    """
    function passed to curve_fit to use gaussian2Drot
    """
    x, y = M
    return gaussian2Drot(x,y,*args)

def absImgNcount(img,isConstrained=False,p0c = None):
    """
    input
        img: a 2D absorption image
        isConstrained: whether to use pre-defined bounds and guesses
        p0c: a tuple of initial guess for x and y parameters

    output: Ncount computed from two 1D Gaussian fits to the data
    """
    t = np.exp(-img);
    q1 = t[5,:];
    q2 = t[-5,:];
    q3 = t[:,5];
    q4 = t[:,-5];
    s = (np.mean(q1+q2) + np.mean(q3+q4))/4
    img = img+np.log(s) # get Norm Ncount
    
    xcut = np.sum(img, axis = 0)
    ycut = np.sum(img, axis = 1)
    
    xmi = np.argmax(xcut)
    ymi = np.argmax(ycut)
    xm = xcut[xmi]
    ym = ycut[ymi]
    
    bs = max(len(xcut),len(ycut))
    
    bound_x = ([xm/10,0,0.0,-1,-0.03],[xm*2,bs,1.5*bs,1,0.03])
    bound_y = ([ym/10,0,0.0,-1,-0.03],[ym*2,bs,1.5*bs,1,0.03])
    p0_x =  [xm,xmi,bs/5,0,0]
    p0_y = [ym,ymi,bs/5,0,0]
    
    if isConstrained and p0c:
        p0_x = p0c[0]
        p0_y = p0c[1]
        bound_x = ((p0_x[0]*0.5,p0_x[1]-1,p0_x[2]*0.95,p0_x[3]-0.2,p0_x[4]-0.001),(p0_x[0]*1.5,p0_x[1]+1,p0_x[2]*1.05,p0_x[3]+0.2,p0_x[4]+0.001))
        bound_y = ((p0_y[0]*0.5,p0_y[1]-1,p0_y[2]*0.95,p0_y[3]-0.2,p0_y[4]-0.001),(p0_y[0]*1.5,p0_y[1]+1,p0_y[2]*1.05,p0_y[3]+0.2,p0_y[4]+0.001))

    fparsX, _ = curve_fit( gaussian_sloped, np.arange(0, len(xcut)), xcut, p0 = p0_x, bounds=bound_x)
    fparsY, _ = curve_fit( gaussian_sloped, np.arange(0, len(ycut)), ycut, p0 = p0_y, bounds=bound_y)
    
    Nx = np.sqrt(2*np.pi) * fparsX[0] * np.abs(fparsX[2])
    Ny = np.sqrt(2*np.pi) * fparsY[0] * np.abs(fparsY[2])
    #return np.sqrt( np.abs(Nx * Ny) )
    return fparsX,fparsY,xcut,ycut

def findCloud(rawimg,window=40):
    """
    input: 2d image
    output: ROI with atom at the center (if it exists)
    """
    meanim = convolve2d(rawimg,np.ones((10,10)),'same')
    xs,ys = meanim.shape
    xmi = np.argmax(np.max(meanim,axis=1,keepdims=0))
    ymi = np.argmax(np.max(meanim,axis=0,keepdims=0))
    if (meanim[xmi,ymi] < 2) or (abs((xmi-xs/2)/(xs/2))>0.55) or (abs((ymi-ys/2)/(ys/2))>0.55):
        flag = 1
        cutout = np.nan
    else:
        cutout = rawimg[xmi-window:xmi+window,ymi-window:ymi+window]
        flag = 0
    return cutout,flag

def fit_background(img):
    """
    input: 2d image
    output: fit parameters for a flat 2D surface a*x + b*y + c
    """
    x = np.arange(0,img.shape[1])
    y = np.arange(0,img.shape[0])
    X, Y = np.meshgrid(x, y)
    
    u = img.flatten();
    no_atom = u<0.1;
    no_atom_2d = np.reshape(no_atom,img.shape)
    no_atom_2d_filt = convolve2d(no_atom_2d,np.ones((3,3)),'same')>4;
    no_atom_2d_filt = no_atom_2d_filt.flatten()
    
    xdata = np.vstack((X.ravel()[no_atom_2d_filt],Y.ravel()[no_atom_2d_filt]));
    p0 = (0,0,0);
    pLo = (-0.1,-0.1,-0.2);
    pHi = (0.1,0.1,0.2);
    pFit,_ = curve_fit(_background2D,xdata,u[no_atom_2d_filt],p0,bounds=(pLo,pHi))
    
    return pFit

def fit_2DGaussian(img):
    """
    input: 2d image
    output: fit parameters for a 2D Gaussian 
    """
    w = medfilt2d(img)
    yc, xc = np.unravel_index(np.argmax(w),w.shape)
    zmax = w[yc,xc]
    
    windowy = np.sum(w[:,xc]>=zmax/2)*3
    windowx = np.sum(w[yc,:]>=zmax/2)*3

    ydist = min(yc,w.shape[0]-yc)
    xdist = min(xc,w.shape[1]-xc)

    windowy = min(windowy,ydist)
    windowx = min(windowx,xdist)
    
    cuty = w[yc-windowy:yc+windowy,xc]
    cutx = w[yc,xc-windowx:xc+windowx]
        
    ys_guess = np.sum(w[:,xc]>=zmax/2)/3
    xs_guess = np.sum(w[yc,:]>=zmax/2)/3
    
    p0x = [zmax,windowx,xs_guess,0]
    p0xLo = (zmax*0.7,windowx-1,xs_guess*0.5,-0.1)
    p0xHi = (zmax*1.1,windowx+1,xs_guess*2,0.1)
    p0y = [zmax,windowy,ys_guess,0]
    p0yLo = (zmax*0.7,windowy-1,ys_guess*0.5,-0.1)
    p0yHi = (zmax*1.1,windowy+1,ys_guess*2,0.1)
    p0x_fit,_ = curve_fit(gaussian,np.arange(0,len(cutx)),cutx,p0x,bounds=(p0xLo,p0xHi))
    p0y_fit,_ = curve_fit(gaussian,np.arange(0,len(cuty)),cuty,p0y,bounds=(p0yLo,p0yHi))
    
    xs = abs(p0x_fit[2])
    ys = abs(p0y_fit[2])
    bgnd = (p0x_fit[3]+p0y_fit[3])/2
    
    cutout = img[yc-windowy:yc+windowy,xc-windowx:xc+windowx]
    y = np.arange(0,cutout.shape[0])
    x = np.arange(0,cutout.shape[1])
    X, Y = np.meshgrid(x, y)
    xdata = np.vstack((X.ravel(), Y.ravel()))
    
    pLo = (windowx-3,windowy-3,xs*0.4,ys*0.4,zmax*0.5,-5*abs(bgnd))
    pHigh = (windowx+3,windowy+3,xs*1.3,ys*1.3,zmax*1.3,5*abs(bgnd))
    p02d = [windowx,windowy,xs,ys,zmax,bgnd]
    
    
    popt,_ = curve_fit(_gaussian2D,xdata,cutout.ravel(),p02d,bounds=(pLo,pHigh))
    return popt,cutout
    
    
    