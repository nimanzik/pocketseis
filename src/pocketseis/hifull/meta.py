from pyrocko.guts import Float, Object


class HIFullMaterial(Object):
    """
    Elastic properties of a homogeneous, isotropic, unbounded
    (full-space) medium.
    Default `vs` value is 5800 m/s (standard crustal value) and vs is
    then set accordingly for a Poisson solid (ν=0.25).
    """

    vp = Float.T(default=5800.0, help="P-wave velocity. Unit: [m/s]")
    vs = Float.T(default=3348.0, help="S-wave velocity. Unit: [m/s]")
    rho = Float.T(default=2600.0, help="Density. Unit: [kg/m^3]")

    def lame(self):
        """
        Lame constants.

        Returns
        -------
        2-tuple of (λ, μ)
        """
        μ = self.vs**2 * self.rho
        λ = self.vp**2 * self.rho - (2.0 * μ)
        return (λ, μ)


__all__ = ["HIFullMaterial"]
