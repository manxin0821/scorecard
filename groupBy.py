
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.DataFrame({'Animal' : ['Falcon', 'Falcon',
                               'Parrot', 'Parrot'],
                   'Max Speed' : [380., 370., 24., 26.]})

a=df.groupby(['Animal']).mean()

