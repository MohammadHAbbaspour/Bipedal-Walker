import matplotlib.pyplot as plt
import numpy as np

class Plot:
    def __init__(self, x, y, z=None, w=None) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def draw2d(self, path='plot.png'):
        plt.plot(self.x, self.y)
        plt.savefig(path)

    def draw3d(self, path='plot.png'):
        import plotly.graph_objects as go
        import plotly
        fig = go.Scatter3d(x=self.x, y=self.y, z=self.z, marker=dict(opacity=1, reversescale=True, colorscale='Blues', size=5, line=dict(width=0.02)), mode='markers')
        layout = go.Layout(scene=dict(xaxis=dict( title="x"),
                                yaxis=dict( title="y"),
                                zaxis=dict(title="xz")),)

        plotly.offline.plot({"data": [fig],
                            "layout": layout},
                            auto_open=True,
                            filename=('plot.html'))

    def draw4d(self, path='plot.png'):
        # from matplotlib import cm
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')        
        # img = ax.scatter(self.x, self.y, self.z, c=self.w, cmap=cm.coolwarm, marker='o')
        # fig.colorbar(img)
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # Plot the surface.
        surf = ax.plot_surface(np.array(self.x), np.array(self.y), np.array(self.z), cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)

        # # Customize the z axis.
        # ax.set_zlim(-1.01, 1.01)
        # ax.zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        ax.zaxis.set_major_formatter('{x:.02f}')

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig(path)


    def save(self, mode, path='plot.png'):
        if mode == '3d':
            self.draw3d(path)
        else:
            self.draw2d(path)