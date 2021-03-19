# Diffusion-fits

Python scripts written by Thomas MacDonald for analysing (time resolved) diffusion NMR data. NMR data is preprocessed in MNova and subsequently used as arrays taken from stacked spectra.

Functions to fit NMR diffusion data in non-overlapping sections, rather than using a moving fit, added by Lucy Fillbrook. Used for processing individual monotonic gradient NMR diffusion experiments that have been stacked, to process all at once rather than individually. Preprocessed NMR data from MNova (integral table) must be input as DataFrame.
