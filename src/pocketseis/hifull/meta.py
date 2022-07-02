from pyrocko.guts import Float, Object


class HifullMaterial(Object):
    """
    Elastic properties of a homogeneous, isotropic, unbounded
    (full-space) medium.
    Default $v_p$ value is 5800 m/s (standard crustal value) and $v_s$
    is then set accordingly for a Poisson solid with $\\nu = 0.25$.
    """
    vp = Float.T(default=5800.0, help='P-wave velocity. Unit: [m/s]')
    vs = Float.T(default=3348.0, help='S-wave velocity. Unit: [m/s]')
    rho = Float.T(default=2600.0, help='Density. Unit: [kg/m^3]')

    def lame(self):
        """
        Lame constants.

        Returns
        -------
        2-tuple of ($\\lambda$, $\\mu$)
        """
        μ = self.vs**2 * self.rho
        λ = self.vp**2 * self.rho - (2.0 * μ)
        return (λ, μ)


__all__ = ['HifullMaterial']
