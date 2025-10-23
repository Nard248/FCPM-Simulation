import argparse
import json

import logging
from dataclasses import dataclass, replace

import numpy as np
import scipy
import scipy.special
from PIL import Image

#######################################################################################
# Load LCSim format.

def lcs_load(filename):
    """
    state_n[x,y,z,d] - director.
    state_alpha[x,y,z] - normal isomer concentration.
    """
    data = np.load(filename)
    settings = json.loads(data['settings'][()])
    state_n = data['PATH'][0]
    state_alpha = data['normalisomer'] if 'normalisomer' in data else data['photo_concentration']
    if len(state_n.shape)==5:
        assert state_n.shape[-2]==1
        state_n = state_n[:,:,:,0]
    if len(state_alpha.shape)==5:
        assert state_alpha.shape[-1]==1
        state_alpha = state_alpha[0,:,:,:,0]
    return settings, state_n, state_alpha

####################################################################################
# Export array to image.

def save_image_bw(filename, data):
    im = Image.fromarray((255*data).astype(np.uint8))
    im.save(filename)

def sgn_to_rgb(data):
    rgb = np.zeros(data.shape+(3,), dtype=np.float32)
    adata = np.abs(data)
    pos=data>0
    rgb[pos,0] = adata[pos]
    neg=data<0
    rgb[neg,2] = adata[neg]
    rgb[...,1] = adata/2 
    return np.clip(rgb, 0, 1)

def save_image_sym(filename, data, mx=None):
    if mx is None:
        mx = np.max(np.abs(data))
    im = Image.fromarray((255*sgn_to_rgb(data/mx)).astype(np.uint8))
    im.save(filename)


###############################################################################################
# Rays

@dataclass
class Rays:
    stepsize:float  # Distance between rays origins.
    wavelength: float 
    E: np.ndarray

    @property
    def shape(self):
        return self.E.shape[:-1]

    def cartesian_coordinates(self):
        sx, sy = self.shape
        rx = sx*self.stepsize/2 # -rx<=x<=rx
        ry = sy*self.stepsize/2 # -ry<=y<=ry
        x = np.linspace( -rx, rx, sx )
        y = np.linspace( -ry, ry, sy )
        xx, yy = np.meshgrid(x, y, indexing='ij')
        return xx, yy
    
    def polar_coordinates(self):
        xx, yy = self.cartesian_coordinates()
        r2 = xx**2+yy**2 # Distance to center.
        r = np.sqrt(r2)
        phi = np.arctan2(yy, xx)
        return r, phi

    @classmethod
    def empty(cls, sx, sy, stepsize, wavelength):
        return cls(
            stepsize=stepsize, 
            wavelength=wavelength, 
            E=np.empty((sx,sy,2), dtype=np.complex64)
        )

    @classmethod
    def uniform(cls, E0, **kwargs):
        beam = cls.empty(**kwargs)
        beam.E[:] = E0
        return beam        

    @classmethod
    def LaguerreGaussianBeam(cls, p, ll, w, E0, **kwargs):
        """ 
            p - radial index
            l - rotational mode number 
            w - radius
            E0 - polarization
        """
        beam = cls.empty(**kwargs)
        r, phi = beam.polar_coordinates()
        r2 = r**2
        # Normalization constant.
        C = np.sqrt(2*scipy.special.factorial(p)/np.pi/scipy.special.factorial(p+np.abs(ll)))

        u = (C/w 
            * np.power(r*np.sqrt(2)/w, np.abs(ll))
            * np.exp(-r2/w**2)
            * scipy.special.assoc_laguerre(2/w**2*r2, n=p, k=np.abs(ll))
            * np.exp(-1j*ll*phi)
            )

        # xy components of electric field.
        beam.E[:] = E0*u[:,:,None]
        return beam

    def intensity(self):
        return np.sum(np.abs(self.E)**2,axis=-1)

    def stokes(self):
        # I=|E_x|^2+|E_y|^2, 
        # Q=|E_x|^2-|E_y|^2, 
        # U=2 Re(E_xE_y^*), 
        # V=-2 Im(E_xE_y^*), 
        Ex2 = np.real( self.E[...,0]*np.conj(self.E[...,0]) )
        Ey2 = np.real( self.E[...,1]*np.conj(self.E[...,1]) )
        ExEy = self.E[...,0]*np.conj(self.E[...,1])

        II = Ex2 + Ey2
        Q = Ex2 - Ey2
        U = 2*np.real(ExEy)
        V = -2*np.imag(ExEy)
        return (II,Q,U,V)


##############################################################################################
# Optic elements

@dataclass
class LinearPolarizer:
    axis: np.ndarray

    @classmethod
    def from_angle(cls, angle_rad:float):
        return cls(
            axis=np.array([np.cos(angle_rad), np.sin(angle_rad)]) 
        )

    def __call__(self, beam):
        if isinstance(beam, Rays):
            return replace(beam,
                E = self.axis*np.sum(self.axis*beam.E, axis=-1)[...,None]
            )


def getindices(x:np.ndarray, s:int, stepsize:float, shift:float):
    """
    For points with coordinates `x` returns indices of the left `i0` and right `i1` closest 
    elements, and local coordinate `l` such that `l=0` corresponds `i0` and `l=1` to `i1`.
    Distance between adjacent elements is given by `stepsize`,
    size of the array by `s` and coordinate of the element `0` by `shift`. 
    """
    x = np.asarray(x)/stepsize+shift
    i0 = x.astype(int)
    i1 = i0+1
    ll = x-i0
    i0 = np.clip(i0, 0, s-1)
    i1 = np.clip(i1, 0, s-1)
    return i0, i1, ll[...,None]

def interpolate(x:np.ndarray, y:np.ndarray, z:np.ndarray, stepsize:float, v:np.ndarray):
    """
    Given vector field `v` return values of the field interpolated to the rectangular grid
    with nodes coordinates given by Cartesian product of `x`, `y` and `z`.
    Center of `v` is assumed to be located at the origin.
    Distance between adjacent elements of `v` is given by `stepsize`. 
    """
    sx,sy,sz,_ = v.shape
    x0, x1, xl = getindices(x, sx, stepsize, sx/2)
    y0, y1, yl = getindices(y, sy, stepsize, sy/2)
    z0, z1, zl = getindices(z, sz, stepsize, 0)
    
    return ( 
        v[x0,y0,z0]*(1-xl)*(1-yl)*(1-zl) + 
        v[x0,y0,z1]*(1-xl)*(1-yl)*(  zl) + 
        v[x0,y1,z0]*(1-xl)*(  yl)*(1-zl) + 
        v[x0,y1,z1]*(1-xl)*(  yl)*(  zl) + 
        v[x1,y0,z0]*(  xl)*(1-yl)*(1-zl) + 
        v[x1,y0,z1]*(  xl)*(1-yl)*(  zl) + 
        v[x1,y1,z0]*(  xl)*(  yl)*(1-zl) + 
        v[x1,y1,z1]*(  xl)*(  yl)*(  zl) 
        )    

@dataclass
class LCFilm:
    thickness: float # Film thickness = size along z coordinate (in microns)
    dz: float # Integration step (um)
    stepsize: float # Distance between adjacent nodes of `m`
    director: np.ndarray # normalized director vector.
    no: float
    ne: float


    def __call__(self, beam):
        if isinstance(beam, Rays):
            return self.apply(beam)

    def apply(self, beam, logger=None):
        if logger is not None: 
            logger.info("Applying LCFilm")
        # Rays coordinates.
        xx, yy = beam.cartesian_coordinates()
        # Electric field.
        E = beam.E.copy()
        # Iterate layers
        z = 0
        stop = False
        while not stop:
            if logger is not None: 
                logger.info(f"Layer {z:3} / {self.thickness:3}")

            # Compute current layer thickness.
            if z+self.dz>self.thickness: # Last layer can have thickness smaller than `dz`.
                dz = self.thickness-z
                stop = True
            else:
                dz = self.dz
            
            # Compute Jones matrices.
            m = interpolate(xx, yy, z+np.zeros_like(xx), self.stepsize, self.director)
            l2 = m[...,0]**2+m[...,1]**2
            ll = np.sqrt(l2)+1e-15
            nx = m[...,0]/ll
            ny = m[...,1]/ll
            p = E[...,0]*nx+E[...,1]*ny
            neff = self.no/np.sqrt( (self.no/self.ne)**2*l2+m[...,2]**2 ) 
            gammaz = np.pi*dz/beam.wavelength*(neff-self.no)
            factor = p*(1-np.cos(2*gammaz)+1j*np.sin(2*gammaz))
            E[...,0] -= factor*nx
            E[...,1] -= factor*ny

            # Go to next layer.
            z = z + dz
        return replace(beam, E=E)



################################################################################################
# Command line interface

def main():
    logging.basicConfig(
        format='%(asctime)s %(module)s[%(levelname)s] %(message)s',
        datefmt='%d.%m %H:%M',
        level=logging.NOTSET)

    numba_logger = logging.getLogger('numba.cuda.cudadrv.driver')
    numba_logger.setLevel(logging.WARNING)

    logger = logging.getLogger()


    ###################################################################################################
    # Parse command line.

    parser = argparse.ArgumentParser(description="Simulate optics of liquid crystal (LC).")
        
    parser.add_argument('state', help='NPZ file containing state of LC.')
    parser.add_argument("--output", '-o', dest='output', default=None, help='File to save output electric field.')
    parser.add_argument("--density", '-d', dest='density', default=None, help='Pixel per micron.', type=int)
    parser.add_argument(
        '--debug', help="Print lots of debugging statements",
        action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.INFO,
    )
    parser.add_argument("--radius", dest='omega', default=10, help='Beam radius (microns).', type=float)
    parser.add_argument("--radial_index", dest='p', default=0, help='Beam radial index p ≥ 0.', type=int)
    parser.add_argument("--azimuthal_index", dest='l', default=0, help='Beam azimuthal index l.', type=int)
    parser.add_argument("--polarizer", '-p', dest='polarizer', default=None, help='Polarizer angle.', type=float)
    parser.add_argument("--analyzer", '-a', dest='analyzer', default=None, help='Analyzer angle.', type=float)
    parser.add_argument("--wavelength", dest='wavelength', default=0.65, help='Wave length (microns).', type=float)
    parser.add_argument("--no", dest='no', default=1.7, help='Ordinary refractive index.', type=float)
    parser.add_argument("--ne", dest='ne', default=1.5, help='Extraordinary refractive index.', type=float)


    args = parser.parse_args()

    # Update verbosity level
    logger.setLevel(args.loglevel)

    #####################################################################################
    # Load NPZ file 

    logger.info(f"Loading {args.state}")
    settings, state_n, _state_alpha = lcs_load(args.state)

    print(f"Settings {settings}")
    print(f"State shape {state_n.shape}")

    #####################################################################################
    # Extract settings.

    thickness = settings['L']
    stepsize = thickness/settings['sz'] # in microns
    sx, sy, sz, _ = state_n.shape
    # lz = sz*stepsize   # 0<=z<=lz

    wavelength = args.wavelength
    no = args.no
    ne = args.ne

    #####################################################################################
    # Generate incident beam.

    # FIXME: normalization?
    # FIXME: control of polarization?
    p=args.p 
    l=args.l 
    w=args.omega
    E0 = np.asarray([1, 1j]) /np.sqrt(2) # Initial polarization.

    beam = beam0 = Rays.LaguerreGaussianBeam(stepsize=stepsize, sx=sx, sy=sy, wavelength=wavelength, 
        p=p, l=l, w=w, E0=E0
        )

    print(f"Beam radial index {p=}, rotational mode number {l=}, radius {w}, polarization {E0}.")

    maxI = np.max(beam0.intensity())
    print(f"Maximum of initial intensity {maxI}.")

    #####################################################################################
    # Apply polarizer.

    if args.polarizer is not None:
        polarizer = LinearPolarizer.from_angle(
            float(args.polarizer)*np.pi/180
        )
        print(f"Applying polarizer with axis {polarizer.axis}")
        beam = polarizer(beam)
    else:
        print("Skipping polarizer")

    #####################################################################################
    # Director interpolation.

    lcf = LCFilm(
        no=no, ne=ne, 
        stepsize=stepsize, director=state_n,
        thickness=thickness, dz=thickness/sz/4
    )
    beam = lcf.apply(beam, logger=logger)

    #####################################################################################
    # Computing Stokes parameters.

    Stokes_I, Stokes_Q, Stokes_U, Stokes_V = beam.stokes()

    #####################################################################################
    # Apply analyzer.

    if args.analyzer is not None:
        analyzer = LinearPolarizer.from_angle(
            float(args.analyzer)*np.pi/180
        )        
        print(f"Applying analyzer with axis {analyzer.axis}")
        beam1 = analyzer(beam)
    else:
        print("Skipping analyzer.")
        beam1 = beam

    # Compute intensity.
    I = beam1.intensity()
    print(f"Maximum final intensity {np.max(I)}")
    I = I/np.max(I)

    #####################################################################################
    # Save results.

    # Make filename.
    name = args.state
    if name[-4:]=='.npz':
        name=name[:-4]

    if args.output is None:
        filename = f"{name}.output.npz"
    else:
        filename = args.output

    if filename[-4:]=='.npz':
        # Save NPZ.
        logger.info(f"Saving electric field to '{filename}'.")
        np.savez_compressed(filename, Einit=beam0.E, Exy=beam1.E, settings=settings,
            Stokes_I=Stokes_I, Stokes_Q=Stokes_Q, Stokes_U=Stokes_U, Stokes_V=Stokes_V
        )
    elif filename[-4:]=='.csv':
        logger.info(f"Saving electric field to '{filename}'.")
        xx, yy = beam.cartesian_coordinates()
        X = np.stack(
            (xx.flatten(), yy.flatten(), 
            Stokes_I.flatten(), Stokes_Q.flatten(), Stokes_U.flatten(), Stokes_V.flatten(),
            np.real(beam0.E[...,0]).flatten(), np.imag(beam0.E[...,0]).flatten(), 
            np.real(beam0.E[...,1]).flatten(), np.imag(beam0.E[...,1]).flatten(), 
            np.real(beam.E[...,0]).flatten(), np.imag(beam.E[...,0]).flatten(), 
            np.real(beam.E[...,1]).flatten(), np.imag(beam.E[...,1]).flatten(), 
            )
            , axis=1)
        np.savetxt(filename, X, fmt='%.18e', delimiter=' ', newline='\n', header='X Y I Q U V rEx0 iEx0 rEy0 iEy0 rEx iEx rEy iEy')

    #####################################################################################
    # Plot images.

    logger.info(f"Saving images to '{name}...'")
    save_image_bw(filename=f"{name}.intensity.png", data=I)

    save_image_sym(filename=f"{name}.Stokes_I.png", data=Stokes_I, mx=maxI/2)
    save_image_sym(filename=f"{name}.Stokes_Q.png", data=Stokes_Q, mx=maxI/2)
    save_image_sym(filename=f"{name}.Stokes_U.png", data=Stokes_U, mx=maxI/2)
    save_image_sym(filename=f"{name}.Stokes_V.png", data=Stokes_V, mx=maxI/2)
