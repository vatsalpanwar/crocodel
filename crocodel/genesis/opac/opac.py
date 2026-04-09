import numpy as np
from .mol_absorp.mol_absorp import Mol_absorp
from .cia.cia import Cia

class Opac:

    def __init__(self):

        self.mol_absorp = Mol_absorp()
        self.cia = Cia()
        
        return

    def opacity(self, atm, include_cia = True):

        self.kappa, self.sigma = self.mol_absorp.opac(atm)
        if include_cia:
            self.kappa += self.cia.opac(atm)

        return
    
    # Added by Vatsal 
    # def opacity_without_opac_check(self, atm):

    #     self.kappa, self.sigma = self.mol_absorp.opac_without_opac_check(atm)
    #     self.kappa += self.cia.opac_without_opac_check(atm)

    #     return