import numpy as np

def cart2pol(xy_arr, center_pos=(0, 0)):
    xy_arr = xy_arr - center_pos
    x = xy_arr[:, 0]
    y = xy_arr[:, 1]
    rho = np.hypot(x, y)  
    phi = np.arctan2(y, x)
    return np.array([rho, phi]).T

def pol2cart(rho_phi_arr, center_pos=(0, 0)):
    rho = rho_phi_arr[:, 0]
    phi = rho_phi_arr[:, 1]
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return np.array([x, y]).T + center_pos

def polar_interpolate(curr_frame_pts, frame_idx=None, non_missing_mask=None, center_pos=None, verbose=True):
    
    if non_missing_mask is None:
        non_missing_mask = ~(curr_frame_pts == 0).all(axis=1)
    
    first_non_missing_idx = np.where(non_missing_mask)[0][0]
    curr_frame_pts = np.roll(curr_frame_pts, -first_non_missing_idx, axis=0)
    non_missing_mask = np.roll(non_missing_mask, -first_non_missing_idx, axis=0)
    non_missing_indices = np.where(non_missing_mask)[0]
    
    if center_pos is None:
        center_pos = curr_frame_pts[non_missing_mask].mean(axis=0)
    
    polar_coords = cart2pol(curr_frame_pts, center_pos)
    polar_coords_deg = polar_coords.copy()
    polar_coords_deg[:, 1] = np.rad2deg(polar_coords_deg[:, 1])
    cart_interpolated_pts = np.zeros_like(curr_frame_pts)
    pt_cnt = len(curr_frame_pts)
    
    for pt_idx in range(pt_cnt):
        if non_missing_mask[pt_idx]:
            cart_interpolated_pts[pt_idx] = curr_frame_pts[pt_idx]
            continue
            
        prev_idx = non_missing_indices[non_missing_indices < pt_idx][-1] if any(non_missing_indices < pt_idx) else non_missing_indices[-1]
        next_idx = non_missing_indices[non_missing_indices > pt_idx][0] if any(non_missing_indices > pt_idx) else non_missing_indices[0]
        
        if next_idx > prev_idx:
            denom = next_idx - prev_idx
        else:
            denom = pt_cnt - (prev_idx - next_idx)
        weight = ((pt_idx - prev_idx) % pt_cnt) / denom
        r1, theta1 = polar_coords[prev_idx]
        r2, theta2 = polar_coords[next_idx]
        if verbose:
            print(f'prev_idx: {prev_idx}, next_idx: {next_idx}')
            print(f'r1: {r1}, theta1: {np.rad2deg(theta1)}')
            print(f'r2: {r2}, theta2: {np.rad2deg(theta2)}')
            print(f'weight: {weight}')

        r_interp = r1 + weight * (r2 - r1)
        theta_diff = (theta2 - theta1)
        if theta_diff < 0:
            theta_diff += 2*np.pi
        theta_interp = theta1 + weight * theta_diff
        if verbose:
            print(f'theta1: {np.rad2deg(theta1)}, theta2: {np.rad2deg(theta2)}')

        cart_interpolated_pts[pt_idx] = pol2cart(np.array([[r_interp, theta_interp]]), center_pos)[0]
        
        if verbose:
            n1, n2 = pol2cart(np.array([[r1, theta1], [r2, theta2]]), center_pos)
            print(f'r_interp: {r_interp}, theta_interp: {np.rad2deg(theta_interp)}')
            print(f'n1: {n1}, n2: {n2}')
            print(f'cart_interpolated_pts: {cart_interpolated_pts[pt_idx]}')

    cart_interpolated_pts = np.roll(cart_interpolated_pts, first_non_missing_idx, axis=0)
    return cart_interpolated_pts
