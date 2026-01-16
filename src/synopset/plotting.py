from __future__ import annotations
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def animate_map(ds: xr.Dataset, var="u10", every=1):
    """Quick viewer to step through time with cluster overlay in title."""
    assert var in ds, f"{var} not in dataset"
    times = pd.DatetimeIndex(ds["time"].values)
    lon = ds["longitude"].values
    lat = ds["latitude"].values
    fig = plt.figure(figsize=(8, 4))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(
        [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())],
        crs=ccrs.PlateCarree(),
    )
    ax.add_feature(
        cfeature.COASTLINE.with_scale("110m"),
        linewidth=0.7,
        edgecolor="black",
        zorder=6,
    )
    cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])

    v0 = ds[var].isel(time=0)
    contour = ax.contourf(
        lon, lat, v0, levels=150, cmap="Spectral_r", transform=ccrs.PlateCarree()
    )
    cb = fig.colorbar(contour, cax=cax)

    lab = ds["cluster_id"] if "cluster_id" in ds else None

    for i in range(0, len(times), every):
        for c in contour.collections:
            c.remove()
        contour = ax.contourf(
            lon,
            lat,
            ds[var].isel(time=i),
            levels=150,
            cmap="Spectral_r",
            transform=ccrs.PlateCarree(),
        )
        cb.update_normal(contour)
        title = str(times[i])
        if lab is not None:
            try:
                cid = int(lab.isel(time=i, latitude=0, longitude=0).values)
            except Exception:
                cid = int(lab.isel(time=i).values)
            title += f" | cluster={cid}"
        ax.set_title(title)
        plt.pause(0.05)
    plt.show()


def probe_panels(
    ds: xr.Dataset,
    wins,
    labels_seq,
    medoid_win_idx,
    probe_lat,
    probe_lon,
    out_png="probe.png",
):
    """Panel per cluster (u10 & v10) at a probe location: all sequences, mean ± std, medoid."""
    assert "u10" in ds and "v10" in ds, "u10 & v10 required"
    lat_name = "latitude"
    lon_name = "longitude"
    u_pt = ds["u10"].sel({lat_name: probe_lat, lon_name: probe_lon}, method="nearest")
    v_pt = ds["v10"].sel({lat_name: probe_lat, lon_name: probe_lon}, method="nearest")
    plat = float(u_pt.coords[lat_name].values)
    plon = float(u_pt.coords[lon_name].values)
    u_series = u_pt.values.astype(float)
    v_series = v_pt.values.astype(float)

    L = wins[0][1] - wins[0][0] + 1
    # infer dt_hours
    times = pd.DatetimeIndex(ds["time"].values)
    dt_hours = float((times[1] - times[0]).total_seconds()) / 3600.0
    t_rel = np.arange(L) * dt_hours
    seq_u = np.array([u_series[s : e + 1] for (s, e) in wins])
    seq_v = np.array([v_series[s : e + 1] for (s, e) in wins])

    k = int(np.max(labels_seq)) + 1
    cluster_members = {c: np.where(labels_seq == c)[0] for c in range(k)}
    nrows, ncols = k, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 2.6 * k), sharex=True)
    if k == 1:
        axes = np.array([axes])
    axes = axes.reshape(k, 2)

    for c in range(k):
        members = cluster_members[c]
        if members.size == 0:
            for j in range(2):
                axes[c, j].set_title(f"Cluster {c} (empty)")
                axes[c, j].axis("off")
            continue
        ax_u = axes[c, 0]
        ax_v = axes[c, 1]
        for m in members:
            ax_u.plot(t_rel, seq_u[m], color="0.7", lw=0.8, alpha=0.5)
            ax_v.plot(t_rel, seq_v[m], color="0.7", lw=0.8, alpha=0.5)
        mu, su = seq_u[members].mean(axis=0), seq_u[members].std(axis=0)
        mv, sv = seq_v[members].mean(axis=0), seq_v[members].std(axis=0)
        ax_u.fill_between(
            t_rel, mu - su, mu + su, alpha=0.15, linewidth=0, label="mean ± std"
        )
        ax_v.fill_between(
            t_rel, mv - sv, mv + sv, alpha=0.15, linewidth=0, label="mean ± std"
        )
        ax_u.plot(t_rel, mu, lw=1.2, alpha=0.9, label="mean")
        ax_v.plot(t_rel, mv, lw=1.2, alpha=0.9, label="mean")
        m_win = medoid_win_idx.get(c, None)
        if m_win is not None:
            ax_u.plot(t_rel, seq_u[m_win], lw=2.3, alpha=0.95, label="medoid")
            ax_v.plot(t_rel, seq_v[m_win], lw=2.3, alpha=0.95, label="medoid")
            s, _ = wins[m_win]
            ts = times[s].strftime("%Y-%m-%d %H:%M")
            ax_u.set_title(f"Cluster {c} | u10 | medoid: {ts}")
            ax_v.set_title(f"Cluster {c} | v10 | medoid window")
        else:
            ax_u.set_title(f"Cluster {c} | u10")
            ax_v.set_title(f"Cluster {c} | v10")
        ax_u.set_ylabel("u10 (m/s)")
        ax_v.set_ylabel("v10 (m/s)")
        ax_u.grid(True, alpha=0.25)
        ax_v.grid(True, alpha=0.25)
        ax_u.legend(loc="upper right", fontsize=8)
        ax_v.legend(loc="upper right", fontsize=8)

    for ax in axes[-1, :]:
        ax.set_xlabel("Hours from window start")
    fig.suptitle(
        f"Probe (lat={plat:.2f}, lon={plon:.2f}) | windows={len(wins)} | k={k}", y=0.999
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"[PLOT] Saved probe plot → {out_png}")
