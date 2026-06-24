import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

df=pd.read_csv('house_prices_dataset.csv')
lr = linear_model.LinearRegression()
lr.fit(df[['home_size','rooms','age', 'distance_from_city(km)' ]], df.price)
home_size, rooms, age, distance = map(float, input("Enter home_size, rooms, age, distance_from_city(km) separated by comma: ").split(','))
result = lr.predict(pd.DataFrame([[home_size, rooms, age, distance]], columns=['home_size','rooms','age','distance_from_city(km)']))
print("Predicted Price:", round(result[0]))
print(lr.coef_)
print(lr.intercept_)
print(lr.score(df[['home_size','rooms','age','distance_from_city(km)']],df.price))

# ── Build Regression Plane ────────────────────────────────────────────────────
x_surf = np.linspace(df['home_size'].min(), df['home_size'].max(), 50)
y_surf = np.linspace(df['rooms'].min(),     df['rooms'].max(),     50)
xx, yy = np.meshgrid(x_surf, y_surf)

X_plane = pd.DataFrame({
    'home_size'              : xx.ravel(),
    'rooms'                  : yy.ravel(),
    'age'                    : df['age'].mean(),
    'distance_from_city(km)' : df['distance_from_city(km)'].mean()
})
zz = lr.predict(X_plane).reshape(xx.shape)

# ── Predicted Point (snapped onto plane) ──────────────────────────────────────
pred_x, pred_y = home_size, rooms
pred_z_plane = lr.predict(pd.DataFrame(
    [[pred_x, pred_y, df['age'].mean(), df['distance_from_city(km)'].mean()]],
    columns=['home_size', 'rooms', 'age', 'distance_from_city(km)']))[0]

# ── 3D Plot ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 8))
ax  = fig.add_subplot(111, projection='3d')

# 1) Plane (draw first)
ax.plot_surface(xx, yy, zz, color='steelblue', alpha=0.3,
                edgecolor='steelblue', linewidth=0.1)

# 2) Scatter points
scatter = ax.scatter(df['home_size'], df['rooms'], df['price'],
                     c=df['age'], cmap='RdYlGn_r',
                     s=df['distance_from_city(km)']*5,
                     alpha=0.5, edgecolors='white', linewidth=0.3)

# 3) Black dot (draw last so it's always on top)
ax.plot([pred_x, pred_x], [pred_y, pred_y], [df['price'].min()*0.9, pred_z_plane],
        color='black', linewidth=1.5, linestyle='--', alpha=0.8)
ax.scatter(pred_x, pred_y, pred_z_plane,
           color='black', s=180, edgecolors='white', linewidth=1.5, depthshade=False)
ax.text(pred_x+80, pred_y+0.1, pred_z_plane+8000,
        f'${round(result[0]):,}\n(rooms={int(rooms)}, age={int(age)})', fontsize=8, fontweight='bold')

# ── Legend — all 4 entries ────────────────────────────────────────────────────
legend_items = [

    # Distance from city (dot size reference)
    *[Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
             markersize=(d * 5) ** 0.5, label=f'{d} km (dist)')
      for d in [5, 15, 25]],

    # Regression plane
    Line2D([0], [0], color='steelblue', linewidth=6, alpha=0.4,
           label='Regression Plane\n(age & dist @ mean)'),

    # Predicted point
    Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
           markersize=9, label=f'Predicted: ${round(result[0]):,}\n'
                               f'({int(home_size)} sqft, {int(rooms)} rooms, age {int(age)}, dist {int(distance)}km)')
]

# Place legend outside the 3D axes so it's never hidden
ax.legend(handles=legend_items, title='Legend',
          loc='upper left', bbox_to_anchor=(0.0, 1.0),
          fontsize=8, framealpha=0.9)

# ── Colorbar for Age ──────────────────────────────────────────────────────────
cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.5)
cbar.set_label('Age (years)', fontsize=10)

# ── Labels ────────────────────────────────────────────────────────────────────
ax.set_xlabel('Home Size (sq ft)', color='red', fontsize=10, labelpad=10)
ax.set_ylabel('Rooms',             color='red', fontsize=10, labelpad=10)
ax.set_zlabel('Price ($)',         color='red', fontsize=10, labelpad=10)
ax.set_title('3D Plot: All 5 Variables + Regression Plane + Predicted Point',
             fontsize=11, fontweight='bold')
ax.view_init(elev=20, azim=135)
plt.tight_layout()
plt.show()