import numpy as np
from enum import Enum, StrEnum
import matplotlib.pyplot as plt



class MSColors(StrEnum):
    GREY = "#7F7F7F"
    LIGHT_BLUE = "#477D9D"
    DARK_BLUE = "#1E3461"
    GREEN = "#5B9276"
    RED = "#D1453D"
    ORANGE = "#E79325"
    # def __str__(self):
    #     return self.value
    
    # def __repr__(self):
    #     return self.value
        
if __name__ == '__main__':
    # print(MSColors.GREY)
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    plt.plot(x, y, c=MSColors.GREY)  # Utilisation de __str__
    plt.title('Graph avec couleur')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()