"""
A collection of functions for working with and plotting NMR data, often
built on top of nmrglue.

BrukerData Class:
Instantiates BrukerData objects, which we can use to work with Bruker
data that has been processed in TopSpin.

UCSFData Class:
Instantiates UCSFData objects, which we can use to work with UCSF
data that has been processed in Sparky.

BMRBData Class:
Consists of methods for retrieving and working with NMR-STAR files

General Methods:
   savesvg()
"""

# Import necessary libraries
import holoviews as hv
from holoviews import opts

hv.extension("matplotlib", logo=False)

import matplotlib.pyplot as plt
import nmrglue as ng
import numpy as np
import pandas as pd
import pynmrstar

# This gives us the cs values as floats instead of strings
pynmrstar.CONVERT_DATATYPES = True

# Let's also declare some useful dictionaries and lists
CC_xpeaks = [
    ["C", "CA"],
    ["CA", "C"],  # GENERAL
    ["CA", "CB"],
    ["CB", "CA"],
    ["CB", "CG"],
    ["CG", "CB"],
    ["CG", "CD"],
    ["CD", "CG"],
    ["CD", "CE"],
    ["CE", "CD"],
    ["CG", "CD1"],
    ["CD1", "CG"],  # TYR, PHE, THR, etc.
    ["CG", "CD2"],
    ["CD2", "CG"],
    ["CD1", "CE1"],
    ["CE1", "CD1"],
    ["CD2", "CE2"],
    ["CE2", "CD2"],
    ["CE1", "CZ"],
    ["CZ", "CE1"],
    ["CB", "CG1"],
    ["CG1", "CB"],  # ILE
    ["CB", "CG2"],
    ["CG2", "CB"],
    ["CG1", "CD1"],
    ["CD1", "CG1"],
    ["CD2", "CE3"],
    ["CE3", "CD2"],  # TRP
    ["CE2", "CZ2"],
    ["CZ2", "CE2"],
    ["CE3", "CZ3"],
    ["CZ3", "CE3"],
    ["CZ2", "CH2"],
    ["CH2", "CZ2"],
    ["CZ2", "CH3"],
    ["CH3", "CZ3"],
]

greeks = {
    "C": "C",
    "CA": "Cα",
    "CB": "Cβ",
    "CG": "Cγ",
    "CD": "Cδ",
    "CE": "Cε",
    "CZ": "Cζ",
    "CH": "Cη",
    "CG1": "Cγ1",
    "CG2": "Cγ2",
    "CD1": "Cδ1",
    "CD2": "Cδ2",
    "CE1": "Cε1",
    "CE2": "Cε2",
    "CE2": "Cε2",
    "CZ2": "Cζ2",
    "CZ3": "Cζ3",
    "CH2": "Cη2",
}


class BrukerData:
    """
    Make an instance of a BrukerData object from processed Bruker data.
    This class uses a few nmrglue methods with all of the arguments set
    to their defaults.

    Parameters
    ----------

    * dir : string
       the parent data directory, "myproject/1/"


    * pdir : string
       the processed data directory,  like "myproject/1/pdata/1/"
    """

    def __init__(self, pdir, rdir):
        self.rdir = rdir
        self.pdir = pdir

        # Instantiate our data object with nmrglue parsers
        self.pdic, self.data = ng.fileio.bruker.read_pdata(pdir, scale_data=True)
        self.dic, _ = ng.fileio.bruker.read(rdir)
        self.udic = ng.fileio.bruker.guess_udic(self.dic, self.data, strip_fake=False)

        # Instantiate a plot object that we can pass to HoloViews
        self.plot = self.get_contours()

    def describe(self):
        """
        Provides a text description of the data
        """
        if len(self.data.shape) == 1:
            x = self.data.shape[0]

            print(f"There are {x} points in our spectrum in total.")
            print(
                f"The largest value is {np.amax(self.data)} and the smallest is {np.amin(self.data)}"
            )

        else:
            x, y = self.data.shape

            print(f"There are {self.data.size} points in our spectrum in total.")
            print(f"The data are contained in a {x} by {y} array.")
            print(
                f"The largest value is {np.amax(self.data)} and the smallest is {np.amin(self.data)}"
            )

    def get_clevels(self):
        """
        Attempts to parse the clevels file from the pdata directory
        specified by init.

        I'm not sure what the spec of this file is so ymmv with the
        parsing.

        Returns
        -------
        A list of the contour levels last saved by TopSpin.
        """

        cfile = f"{self.pdir}/clevels"

        clevels = []

        with open(cfile, "rt") as f:
            for line in f:
                if line[0].isdigit() or line[0] == "-":
                    for entry in line.split():
                        if entry != "0":
                            try:
                                clevels.append(float(entry))
                            except ValueError:
                                pass
                        else:
                            continue

        return clevels

    def get_contours(self, scale=True, **kwargs):
        """
        Builds contours for Bruker data using the QuadContourSet object from matplotlib.

        Parameters
        ----------
        * scale : bool
           If True, converts the axis scaling to ppm.

        * clevels : list (optional)
           A list corresponding to the contour levels. If no list is
           provided, the contour levels are calculated using get_clevels()

        Returns
        -------
        A list of dictionaries of contour lines and their corresponding
        contour level. This can be passed directly to a plotting library
        like HoloViews to build NMR plots.
        """

        # Builds a plottable object for 1D data. This is just a 2D list with ppm - intensity pairs.
        # This also skips the clevels call.
        if len(self.data.shape) == 1:
            if scale:
                ppms = self.get_ppm()[0]
                data = self.data

                conts = []
                for intensities, cshifts in zip(data, ppms):
                    point = [cshifts, intensities]
                    conts.append(point)
            else:
                conts = self.data

            return conts

        # Determine contour levels for this spectrum, or use levels
        # defined in the method call.
        if "clevels" in kwargs:
            clevels = kwargs.get("clevels")
        else:
            clevels = self.get_clevels()

        # Build meshgrids for ppm values and adjust the referencing to
        # match the referencing defined in the procpars file.
        # Note that the file naming approach taken by Bruker can vary
        # between different versions of TopSpin so this function may
        # complain if an older version of TopSpin was used to process the
        # data.
        if scale:
            ppm_vectors = []

            for dim in np.arange(0, self.udic["ndim"]):
                # nmrglue stumbles here with newer versions of TopSpin. We
                # need to set these values manually to reference our
                # spectrum.
                self.udic[dim]["obs"] = self.pdic["procs"]["SF"]
                self.udic[dim]["car"] = (
                    self.dic["acqus"]["SFO1"] - self.udic[dim]["obs"]
                ) * 1e6

                # We can now use nmrglue's built-in conversion methods to
                # build direct and indirect ppm vectors to build our
                # meshgrids.
                uc = ng.fileiobase.uc_from_udic(self.udic, dim)
                ppmsc = uc.ppm_scale()
                ppm_vectors.append(ppmsc)

            # Build the meshgrids. This only works in 2D for now.
            xgrid, ygrid = np.meshgrid(ppm_vectors[1], ppm_vectors[0])

            # Use plt to calculate contours
            mycnts = plt.contour(xgrid, ygrid, self.data, levels=clevels)
            # stop this from showing the matplotlib plot
            plt.close()
        else:
            # Use plt to calculate contours
            mycnts = plt.contour(self.data, levels=clevels)
            # stop this from showing the matplotlib plot
            plt.close()

        cdictlist = []
        level = 0

        # Parse plt's allsegs object to extract contour line coordinates
        for seg in mycnts.allsegs:
            level_value = clevels[level]
            level += 1

            for poly in seg:
                cdict = {("x", "y"): poly, "level": level_value}
                cdictlist.append(cdict)

        return cdictlist

    def get_ppm(self):
        """
        Counts the number of dimensions and returns ppm scales for
        each dimension

        Returns
        -------
        A list of scalars.
        """

        ppms = []

        for dim in np.arange(0, self.udic["ndim"]):
            # nmrglue stumbles here with newer versions of TopSpin. We
            # need to set these values manually to reference our
            # spectrum.
            self.udic[dim]["obs"] = self.pdic["procs"]["SF"]
            self.udic[dim]["car"] = (
                self.dic["acqus"]["SFO1"] - self.udic[dim]["obs"]
            ) * 1e6

            # We can now use nmrglue's built-in conversion methods to
            # build direct and indirect ppm vectors to build our
            # meshgrids.
            uc = ng.fileiobase.uc_from_udic(self.udic, dim)
            ppmsc = uc.ppm_scale()
            ppms.append(ppmsc)

        return ppms


class UCSFData:
    """
    Make an instance of a UCSFData object from processed UCSF data.
    This class uses a few nmrglue methods with all of the arguments set
    to their defaults.

    Parameters
    ----------

    * file : string
       the file to be used, "mydata.ucsf"

    """

    def __init__(self, file):
        self.filename = file
        self.file = ng.fileio.sparky.read(self.filename)

        # Instantiate our data object with nmrglue parsers
        self.data = self.file[1]

    def describe(self):
        """
        Provides a text description of the data
        """

        x, y = self.data.shape

        print(f"There are {self.data.size} points in our spectrum in total.")
        print(f"The data are contained in a {x} by {y} array.")
        print(
            f"The largest value is {np.amax(self.data)} and the smallest is {np.amin(self.data)}"
        )

    def get_clevels(self, clevels):
        """
        It isn't clear where the contour levels are saved by Sparky, so this method will just set the
        clevels parameter for the UCSFData class as defined by the user. These values can be copied
        from Sparky directly or calculated from the min, num, and factor values as follows:

        clevels = contour_min * contour_factor ** np.arange(contour_num)
        clevels = np.where(clevels==0, contour_min, clevels)
        """

        self.clevels = clevels

    def get_contours(self, scale=True, **kwargs):
        """
        Builds contours for UCSF data using the QuadContourSet object from matplotlib.

        Parameters
        ----------
        * clevels : list (optional)
           A list corresponding to the contour levels. If no list is
           provided, the contour levels are calculated using get_clevels().

        Returns
        -------
        A list of dictionaries of contour lines and their corresponding
        contour level. This can be passed directly to a plotting library
        like HoloViews to build NMR plots.
        """

        # Determine contour levels for this spectrum, or use levels
        # defined in the method call.
        if "clevels" in kwargs:
            clevels = kwargs.get("clevels")
        else:
            try:
                clevels = self.clevels()
            except:
                print("Contour levels have not been set!")

        # Build meshgrids for ppm values and generate contours. Only works for 2D data.
        if scale:
            # Convert axes
            uc0 = ng.fileio.sparky.make_uc(self.file[0], self.file[1], dim=0)
            uc1 = ng.fileio.sparky.make_uc(self.file[0], self.file[1], dim=1)

            ppmsc0 = uc0.ppm_scale()
            ppmsc1 = uc1.ppm_scale()

            xgrid, ygrid = np.meshgrid(ppmsc1, ppmsc0)

            # Get contours
            mycnts = plt.contour(xgrid, ygrid, self.data, levels=clevels)
            plt.close()

        else:
            # Use plt to calculate contours
            mycnts = plt.contour(self.data, levels=clevels)
            # stop this from showing the matplotlib plot
            plt.close()

        cdictlist = []
        level = 0

        # Parse plt's allsegs object to extract contour line coordinates
        for seg in mycnts.allsegs:
            level_value = clevels[level]
            level += 1

            for poly in seg:
                cdict = {("x", "y"): poly, "level": level_value}
                cdictlist.append(cdict)

        return cdictlist

    def get_ppm(self):
        """
        Counts the number of dimensions and returns the minimum and
        maximum values in ppm for each axis.

        Returns
        -------
        A list of scalars.
        """
        uc0 = ng.fileio.sparky.make_uc(self.file[0], self.file[1], dim=0)
        uc1 = ng.fileio.sparky.make_uc(self.file[0], self.file[1], dim=1)

        ppmsc0 = uc0.ppm_scale()
        ppmsc1 = uc1.ppm_scale()

        ppms = [ppmsc0, ppmsc1]

        return ppms


class BMRBData:
    """
    Make an instance of a BMRBData object using a BMRB ID and methods
    from NMRFAM's pynmrstar library

    Parameters
    ----------

    * dir : int
       the BMRB ID
    """

    def __init__(self, id):
        self.id = id

        # Get the NMR-STAR file from the BMRB
        self.nmrstar = pynmrstar.Entry.from_database(self.id)

        # Extract the chemical shifts. This can be done in __init__.
        self.cs = []
        for chemical_shift_loop in self.nmrstar.get_loops_by_category(
            "Atom_chem_shift"
        ):
            self.cs.append(
                chemical_shift_loop.get_tag(
                    ["Comp_index_ID", "Comp_ID", "Atom_ID", "Val"]
                )
            )

        csdf = pd.DataFrame(self.cs[0])
        csdf.columns = ["INDEX", "RESIDUE_TYPE", "ATOM_NAME", "CS"]
        self.csdf = csdf

    def get_xpeak_df(self):
        """
        Generates a dataframe of chemical shift crosspeaks

        Returns
        -------
        A dataframe containing theoretical chemical shift crosspeaks
        """
        #

        gcsdf = self.csdf.groupby("INDEX")
        xpeak_df = pd.DataFrame(
            columns=[
                "Index",
                "Residue",
                "Direct",
                "Direct_CS",
                "Indirect",
                "Indirect_CS",
            ]
        )

        # cycle through all of the residues in our grouped df
        for residue in gcsdf:
            # cycle through the whole xpeaks list and match correlations
            for cors in CC_xpeaks:
                direct_atom, indirect_atom = cors[0], cors[1]
                if (
                    direct_atom in residue[1].ATOM_NAME.values
                    and indirect_atom in residue[1].ATOM_NAME.values
                ):
                    index = residue_type = residue[1]["INDEX"].iloc[0]
                    residue_type = residue[1]["RESIDUE_TYPE"].iloc[0]
                    x_coord = residue[1][residue[1]["ATOM_NAME"] == direct_atom][
                        "CS"
                    ].item()
                    y_coord = residue[1][residue[1]["ATOM_NAME"] == indirect_atom][
                        "CS"
                    ].item()

                    xpeak_df = xpeak_df.append(
                        {
                            "Index": index,
                            "Residue": residue_type,
                            "Direct": greeks[direct_atom],
                            "Direct_CS": x_coord,
                            "Indirect": greeks[indirect_atom],
                            "Indirect_CS": y_coord,
                        },
                        ignore_index=True,
                    )

        return xpeak_df


"""
General Methods
"""


def savesvg(plot, filepath, *args, **kwargs):
    # TODO: Implement hooks here.
    # br = hv.renderer('matplotlib')
    # svg = br.get_plot(plot)
    # svg = svg.state
    # svg.output_backend='svg'
    hv.save(plot, filepath)
