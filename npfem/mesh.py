import gmsh
import numpy as np
import torch
from npfem.global_module import g  # g is the global instance containing all data
from scipy import sparse


class GmshMesher:
    """
    Delaunay + alpha-shape using Gmsh.
    2D and 3D support.
    """

    def __init__(self, dim=2, mesh_size=0.01, alpha=None, refine=False, rmin_factor=0.2, rmax_factor=3.0):
        if dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")

        self.dim = dim
        self.mesh_size = mesh_size
        self.smax = mesh_size
        self.smin = rmin_factor * mesh_size
        self.dmax = rmax_factor * self.smax
        self.alpha = alpha if alpha is not None else (1.2 if dim == 2 else 3.0)
        self.refine = refine

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)

        # --- Geometry setup ---
        if dim == 2:
            gmsh.model.add("ModelGeo")
            gmsh.model.occ.addRectangle(-1, -1, 0, 2, 2)
            gmsh.model.occ.synchronize()
            gmsh.model.mesh.generate(2)
            bnd_entities = gmsh.model.getEntities(1)

            gmsh.model.add("ModelAlpha")
            for d, tag in bnd_entities:
                gmsh.model.addDiscreteEntity(d, tag)
            self.domainTag = gmsh.model.addDiscreteEntity(2, -1, [])
            self.boundaryTag = gmsh.model.addDiscreteEntity(1, -1, [])

            gmsh.model.addPhysicalGroup(1, [self.boundaryTag], -1, "freeSurface")
            gmsh.model.addPhysicalGroup(2, [self.domainTag], -1, "domain")

        if dim == 3:
            gmsh.model.add("ModelGeo")
            gmsh.model.occ.addBox(-1, -1, 0, 2, 2, 1)
            gmsh.model.occ.synchronize()
            gmsh.model.mesh.generate(2)
            bnd_entities = gmsh.model.getEntities(2)

            gmsh.model.add("ModelFluid")
            for d, tag in bnd_entities:
                gmsh.model.addDiscreteEntity(d, tag)
            self.asSurfaceTag = gmsh.model.addDiscreteEntity(2, -1)
            self.asVolumeTag = gmsh.model.addDiscreteEntity(3, -1, [self.asSurfaceTag])
            self.domainTag = gmsh.model.addDiscreteEntity(3, -1)

            gmsh.model.addPhysicalGroup(2, [self.asSurfaceTag], -1, "alphaShape_surface")
            gmsh.model.addPhysicalGroup(3, [self.asVolumeTag], -1, "alphaShape_volume")

        # --- Size fields ---
        if dim == 2:
            self.sizeFieldConstant = gmsh.model.mesh.field.add("Box")
            gmsh.model.mesh.field.setNumber(self.sizeFieldConstant, "VIn", mesh_size)
            gmsh.model.mesh.field.setNumber(self.sizeFieldConstant, "VOut", mesh_size if dim == 3 else 10 * mesh_size)
            gmsh.model.mesh.field.setNumber(self.sizeFieldConstant, "XMin", 0.0)
            gmsh.model.mesh.field.setNumber(self.sizeFieldConstant, "XMax", 1.0)
            gmsh.model.mesh.field.setNumber(self.sizeFieldConstant, "YMin", 0.0)
            gmsh.model.mesh.field.setNumber(self.sizeFieldConstant, "YMax", 1.0)
            gmsh.model.mesh.field.setNumber(self.sizeFieldConstant, "Thickness", 0.001)
        if dim == 3:
            self.sizeFieldConstant = gmsh.model.mesh.field.add("Box")
            gmsh.model.mesh.field.setNumber(self.sizeFieldConstant, "VIn", mesh_size)
            gmsh.model.mesh.field.setNumber(self.sizeFieldConstant, "VOut", mesh_size if dim == 3 else 10 * mesh_size)
            gmsh.model.mesh.field.setNumber(self.sizeFieldConstant, "XMin", -1.0)
            gmsh.model.mesh.field.setNumber(self.sizeFieldConstant, "XMax", 1.0)
            gmsh.model.mesh.field.setNumber(self.sizeFieldConstant, "YMin", -1.0)
            gmsh.model.mesh.field.setNumber(self.sizeFieldConstant, "YMax", 1.0)
            gmsh.model.mesh.field.setNumber(self.sizeFieldConstant, "ZMin", 0.0)
            gmsh.model.mesh.field.setNumber(self.sizeFieldConstant, "ZMax", 1.0)
            gmsh.model.mesh.field.setNumber(self.sizeFieldConstant, "Thickness", 0.001)

        if self.refine:
            if dim == 2:
                self.sizeFieldDist = gmsh.model.mesh.field.add("AlphaShapeDistance")
                gmsh.model.mesh.field.setNumber(self.sizeFieldDist, "Tag", self.boundaryTag)
                gmsh.model.mesh.field.setNumber(self.sizeFieldDist, "SamplingLength", 0.25 * self.smin)
            if dim == 3:
                self.sizeFieldDist = gmsh.model.mesh.field.add("AlphaShapeDistance")
                gmsh.model.mesh.field.setNumber(self.sizeFieldDist, "Tag", self.asSurfaceTag)
                gmsh.model.mesh.field.setNumber(self.sizeFieldDist, "Dim", 3)
                gmsh.model.mesh.field.setNumber(self.sizeFieldDist, "VolumeTag", self.asVolumeTag)
                gmsh.model.mesh.field.setNumber(self.sizeFieldDist, "SamplingLength", 0.2 * self.smin)

            self.sizeFieldRefine = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(self.sizeFieldRefine, "InField", self.sizeFieldDist) # which field to use to compute distance
            gmsh.model.mesh.field.setNumber(self.sizeFieldRefine, "SizeMin", self.smin)
            gmsh.model.mesh.field.setNumber(self.sizeFieldRefine, "SizeMax", self.smax)
            gmsh.model.mesh.field.setNumber(self.sizeFieldRefine, "DistMin", 0.0)
            gmsh.model.mesh.field.setNumber(self.sizeFieldRefine, "DistMax", self.dmax)

    # --- Meshing API ---
    def first_mesh(self, position_np: np.ndarray):
        gmsh.model.mesh.clear([(self.dim, self.domainTag)])
        if self.dim == 2:
            coords3d = np.hstack([position_np, np.zeros((position_np.shape[0], 1))])
            gmsh.model.mesh.addNodes(2, self.domainTag, np.arange(1, position_np.shape[0] + 1), coords3d.flatten())
            gmsh.model.mesh.computeAlphaShape(2, self.domainTag, self.boundaryTag,
                                              "ModelGeo", self.alpha,
                                              self.sizeFieldConstant, self.sizeFieldConstant,
                                              False, boundaryTolerance=0.005 * self.smin,
                                              refine=False, deleteDisconnectedNodes=False)
            elem_type = 2
        else:
            gmsh.model.mesh.addNodes(3, self.domainTag, [], position_np.flatten())
            gmsh.model.mesh.tetrahedralizePoints(self.domainTag, False)
            gmsh.model.mesh.alphaShape3D(self.domainTag, self.alpha,
                                         self.sizeFieldConstant, self.asVolumeTag, self.asSurfaceTag)
            elem_type = 4

        tags, coords, _ = gmsh.model.mesh.getNodes(self.dim, self.domainTag)
        if self.dim == 2:
            _, cells_np = self.getElements(self.domainTag, 2, tags)
            cells_np = cells_np.reshape((-1, 3))
        else:    
            _, cells_np = self.getElements(self.asVolumeTag, elem_type, tags)
            cells_np = cells_np.reshape((-1, 4))
        coords = coords.reshape((-1, 3))
        return cells_np, coords[:, :self.dim], tags

    def generate_mesh(self, position_np: np.ndarray, tags: np.ndarray, refine: bool=False):
        gmsh.model.mesh.clear([(self.dim, self.domainTag)])
        if self.dim == 2:
            coords3d = np.hstack([position_np, np.zeros((position_np.shape[0], 1))])
            gmsh.model.mesh.addNodes(2, self.domainTag, tags, coords3d.flatten())
            if refine:
                gmsh.model.mesh.computeAlphaShape(2, self.domainTag, self.boundaryTag,
                                              "ModelGeo", self.alpha,
                                              self.sizeFieldConstant, self.sizeFieldRefine,
                                              False, boundaryTolerance=0.005 * self.smin,
                                              refine=refine, deleteDisconnectedNodes=True)
            else:
                gmsh.model.mesh.computeAlphaShape(2, self.domainTag, self.boundaryTag,
                                              "ModelGeo", self.alpha,
                                              self.sizeFieldConstant, self.sizeFieldConstant,
                                              False, boundaryTolerance=0.005 * self.smin,
                                              refine=False, deleteDisconnectedNodes=True)
            elem_type = 2
        else:
            gmsh.model.mesh.addNodes(3, self.domainTag, tags, position_np.flatten())
            gmsh.model.mesh.tetrahedralizePoints(self.domainTag, False)
            gmsh.model.mesh.alphaShape3D(self.domainTag, self.alpha,
                                         self.sizeFieldConstant, self.asVolumeTag, self.asSurfaceTag)
            if self.refine:
                gmsh.model.mesh.volumeMeshRefinement(self.domainTag, self.asSurfaceTag, self.asVolumeTag, self.sizeFieldRefine, False)
            elem_type = 4

        tags, coords, _ = gmsh.model.mesh.getNodes(self.dim, self.domainTag)
        if self.dim == 2:
            _, cells_np = self.getElements(self.domainTag, elem_type, tags)
            cells_np = cells_np.reshape((-1, 3))
        else:    
            _, cells_np = self.getElements(self.asVolumeTag, elem_type, tags)
            cells_np = cells_np.reshape((-1, 4))
        coords = coords.reshape((-1, 3))
        return cells_np, coords[:, :self.dim], tags

    # --- Helpers ---
    def getElements(self, tag, elem_type, ntDomain):
        _, elementsDomain = gmsh.model.mesh.getElementsByType(elem_type, tag)
        elementsDomain = np.searchsorted(ntDomain, elementsDomain)
        return _, elementsDomain

    def finalize_mesher(self):
        gmsh.finalize()
    
    def get_prev_mesh(self):
        """Returns the previous mesh elements and nodes."""
        tags, coords, _ = gmsh.model.mesh.getNodes(self.dim, self.domainTag)
        if self.dim == 2:
            _, cells_np = self.getElements(self.domainTag, 2, tags)
            cells_np = cells_np.reshape((-1, 3))
        else:    
            _, cells_np = self.getElements(self.asVolumeTag, 4, tags)
            cells_np = cells_np.reshape((-1, 4))
        return cells_np
    

    def calculate_shape_functions(self, elem_coords: torch.Tensor, point: torch.Tensor):
        """
        elem_coords: [K, D] (triangle or tetrahedron node coordinates)
        point: [D]   target point
        Returns: [K] barycentric coordinates (shape functions)
        """
        D = elem_coords.shape[1]
        K = elem_coords.shape[0]

        if D == 2 and K == 3:
            # --- Triangle (2D) ---
            x1, y1 = elem_coords[0]
            x2, y2 = elem_coords[1]
            x3, y3 = elem_coords[2]
            x, y   = point

            det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
            N1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / det
            N2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / det
            N3 = 1.0 - N1 - N2
            return torch.tensor([N1, N2, N3], dtype=elem_coords.dtype, device=elem_coords.device)

        elif D == 3 and K == 4:
            # --- Tetrahedron (3D) ---
            v0, v1, v2, v3 = elem_coords

            def volume(a, b, c, d):
                return torch.abs(torch.det(torch.stack([b - a, c - a, d - a], dim=-1))) / 6.0

            vol_total = volume(v0, v1, v2, v3)
            N0 = volume(point, v1, v2, v3) / vol_total
            N1 = volume(v0, point, v2, v3) / vol_total
            N2 = volume(v0, v1, point, v3) / vol_total
            N3 = volume(v0, v1, v2, point) / vol_total
            return torch.stack([N0, N1, N2, N3])

        else:
            raise ValueError(f"Unsupported element shape with {K} nodes in {D}D")
    
    def find_elements(self, points: torch.Tensor, nodes: torch.Tensor, elements: torch.Tensor):
        """
        Vectorized element finder for 2D (triangles) and 3D (tetrahedra).

        Args:
            points:   [N, D] query points
            nodes:    [N_nodes, D] node positions
            elements: [M, K] element connectivity (K=3 tri, K=4 tet)

        Returns:
            elem_idx: [N] element index for each point (-1 if not inside any)
            bary:     [N, K] barycentric coords (valid only if inside)
        """
        device = points.device
        D = points.shape[1]
        K = elements.shape[1]
        assert (D == 2 and K == 3) or (D == 3 and K == 4), "Only supports tri(2D)/tet(3D)."

        # Gather element node coords [M, K, D]
        elem_coords = nodes[elements]  # [M, K, D]

        # Expand dims for broadcasting
        pts = points[:, None, :]       # [N, 1, D]
        elems = elem_coords[None, :, :, :]  # [1, M, K, D]

        if D == 2:
            # --- Triangles ---
            A = elems[:, :, 0, :]  # [1, M, 2]
            B = elems[:, :, 1, :]
            C = elems[:, :, 2, :]
            P = pts

            detT = (B[..., 1] - C[..., 1]) * (A[..., 0] - C[..., 0]) + \
                (C[..., 0] - B[..., 0]) * (A[..., 1] - C[..., 1])  # [1, M]

            l1 = ((B[..., 1] - C[..., 1]) * (P[..., 0] - C[..., 0]) + \
                (C[..., 0] - B[..., 0]) * (P[..., 1] - C[..., 1])) / detT
            l2 = ((C[..., 1] - A[..., 1]) * (P[..., 0] - C[..., 0]) + \
                (A[..., 0] - C[..., 0]) * (P[..., 1] - C[..., 1])) / detT
            l3 = 1.0 - l1 - l2

            bary = torch.stack([l1, l2, l3], dim=-1)  # [N, M, 3]

            inside = (bary >= -1e-8).all(dim=-1) & (bary <= 1+1e-8).all(dim=-1)  # [N, M]

        else:
            # --- Tetrahedra ---
            A = elems[:, :, 0, :]  # [1, M, 3]
            B = elems[:, :, 1, :]
            C = elems[:, :, 2, :]
            Dv = elems[:, :, 3, :]
            P = pts

            def tet_vol(a, b, c, d):
                return torch.sum(torch.cross(b - a, c - a, dim=-1) * (d - a), dim=-1)

            volT = tet_vol(A, B, C, Dv)  # [1, M]

            v1 = tet_vol(P, B, C, Dv) / volT
            v2 = tet_vol(A, P, C, Dv) / volT
            v3 = tet_vol(A, B, P, Dv) / volT
            v4 = tet_vol(A, B, C, P) / volT

            bary = torch.stack([v1, v2, v3, v4], dim=-1)  # [N, M, 4]

            inside = (bary >= -1e-8).all(dim=-1) & (bary <= 1+1e-8).all(dim=-1)  # [N, M]

        # Pick first valid element per point
        elem_idx = torch.full((points.shape[0],), -1, device=device, dtype=torch.long)
        bary_out = torch.zeros((points.shape[0], K), device=device, dtype=points.dtype)

        any_inside = inside.any(dim=1)  # [N]
        idx_inside = any_inside.nonzero(as_tuple=True)[0]

        if idx_inside.numel() > 0:
            first_hit = inside[idx_inside].float().argmax(dim=1)  # pick first element index
            elem_idx[idx_inside] = first_hit
            bary_out[idx_inside] = bary[idx_inside, first_hit, :]

        return elem_idx, bary_out
    
    
    def triangle_area_vec(self,x1, y1, x2, y2, x3, y3):
        """Vectorized triangle area (absolute)."""
        return np.abs((x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)) / 2.0

    def triangle_area_single(self, p0, p1, p2):
        """Single triangle area (absolute)."""
        x1, y1 = p0; x2, y2 = p1; x3, y3 = p2
        return abs((x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)) / 2.0

    def barycentric_triangle(self, elem_coords, point, tol=1e-15):
        """
        Return barycentric coordinates [N1,N2,N3] for a 2D triangle.
        elem_coords: (3,2) array; point: (2,) array
        Raises ValueError for degenerate triangle.
        """
        x1, y1 = elem_coords[0]
        x2, y2 = elem_coords[1]
        x3, y3 = elem_coords[2]
        x, y = point
        det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if abs(det) <= tol:
            raise ValueError("Degenerate triangle (zero determinant)")
        N1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / det
        N2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / det
        N3 = 1.0 - N1 - N2
        return np.array([N1, N2, N3], dtype=float)
    
    def addrem_nodes_local(self):
        """
        Vectorized AddRemNodesLocal using global object g.
        Operates in-place on g.coords, g.V_tn, g.P_tn, g.strain_el, etc.
        """
        npneigh = g.neighb.cpu().numpy()
        npoints = g.coords.shape[0]
        nelement = g.cells.shape[0]

        # --- 1) compute element areas & mean radius ---
        n1, n2, n3 = g.cells[:, 0], g.cells[:, 1], g.cells[:, 2]
        x1, y1 = g.coords[n1, 0], g.coords[n1, 1]
        x2, y2 = g.coords[n2, 0], g.coords[n2, 1]
        x3, y3 = g.coords[n3, 0], g.coords[n3, 1]

        areas = self.triangle_area_vec(x1, y1, x2, y2, x3, y3)
        area_m = areas.mean()
        radius_mean = np.sqrt(2.0 * area_m) / 2.0

        # --- 2) compute sides & incircle radius ---
        l1 = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        l2 = np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
        l3 = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
        perim = l1 + l2 + l3
        with np.errstate(divide="ignore", invalid="ignore"):
            rad_in = 2.0 * areas / perim

        # --- 3) bad element mask using incircle criterion ---
        bad_elem = (rad_in < 0.5 * radius_mean) | (rad_in > 1.5 * radius_mean)

        # --- 4) skip elements with any free-surface node ---
        elem_has_free = (g.free_surf[n1] == 1) | (g.free_surf[n2] == 1) | (g.free_surf[n3] == 1)
        bad_elem &= ~elem_has_free

        # --- 5) choose candidate node for each bad element ---
        candidate_node = -np.ones(nelement, dtype=int)

        def node_ok(node_idx_arr):
            return (g.free_surf[node_idx_arr] == 0) & (np.any(g.boundv[node_idx_arr] == 0)) & (g.euler[node_idx_arr] == 0)
        mask1 = bad_elem & (l1 < l3) & (l2 < l3) & node_ok(n1)
        candidate_node[mask1] = n1[mask1]

        mask2 = bad_elem & (l1 < l2) & (l3 < l2) & node_ok(n2)
        candidate_node[mask2] = n2[mask2]

        mask3 = bad_elem & (l2 < l1) & (l3 < l1) & node_ok(n3)
        candidate_node[mask3] = n3[mask3]

        fallback = bad_elem & (candidate_node == -1)
        fb1 = fallback & node_ok(n1); candidate_node[fb1] = n1[fb1]
        fb2 = fallback & (candidate_node == -1) & node_ok(n2); candidate_node[fb2] = n2[fb2]
        fb3 = fallback & (candidate_node == -1) & node_ok(n3); candidate_node[fb3] = n3[fb3]

        # --- 6) unique list of nodes to move ---
        valid = candidate_node[candidate_node >= 0]
        if valid.size == 0:
            return

        move_nodes = np.unique(valid)
        moved_nodes_list = []
        count_nodes = 0
        made_addrem_flag = False

        # --- Precompute node -> elements mapping ---
        node_to_elem = [[] for _ in range(npoints)]
        for ei in range(nelement):
            a, b, c = g.cells[ei]
            node_to_elem[a].append(ei)
            node_to_elem[b].append(ei)
            node_to_elem[c].append(ei)

        tol = 1e-12
        for ni in move_nodes:
            ni = int(ni)
            if g.free_surf[ni] == 1 or np.any(g.boundv[ni] == 1) or g.euler[ni] == 1:
                continue

            neigh_ids = np.array(npneigh[ni, 1:])
            valid_neigh = neigh_ids[neigh_ids >= 0]
            if valid_neigh.size == 0:
                continue

            Xg = g.coords[valid_neigh, 0].mean()
            Yg = g.coords[valid_neigh, 1].mean()
            centroid = np.array([Xg, Yg], dtype=float)

            candidate_elems = node_to_elem[ni]
            if len(candidate_elems) == 0:
                continue

            elem_scelto = None
            for ei in candidate_elems:
                tri_nodes = g.cells[ei]
                tri_coords = g.coords[tri_nodes]
                area_el = self.triangle_area_single(tri_coords[0], tri_coords[1], tri_coords[2])
                area1 = self.triangle_area_single(centroid, tri_coords[1], tri_coords[2])
                area2 = self.triangle_area_single(tri_coords[0], centroid, tri_coords[2])
                area3 = self.triangle_area_single(tri_coords[0], tri_coords[1], centroid)
                if area1 + area2 + area3 <= area_el + tol:
                    elem_scelto = ei
                    break

            if elem_scelto is None:
                continue

            tri_nodes = g.cells[elem_scelto]
            tri_coords = g.coords[tri_nodes]
            try:
                L = self.barycentric_triangle(tri_coords, centroid)
            except ValueError:
                continue

            # Move node
            g.coords[ni, 0] = Xg
            g.coords[ni, 1] = Yg
            #if g.solver == 1:
            #if hasattr(g, "V_tn") and g.V_tn is not None:
            #g.V_tn[ni] = L.dot(g.V_tn[tri_nodes])
            #if hasattr(g, "P_tn") and g.P_tn is not None:
            #g.P_tn[ni] = L.dot(g.P_tn[tri_nodes])
            #made_addrem_flag = True

            moved_nodes_list.append(ni)
            count_nodes += 1
        return g.coords

    def compute_node_neighbors(self,elements, n_points=None, max_neighbors=15):
        


        elements = torch.as_tensor(elements, dtype=torch.long, device="cuda")

        if n_points is None:
            n_points = torch.max(elements).item() + 1

        # ----------------------------------------------------------
        # 1) Build edges from triangle connectivity
        # ----------------------------------------------------------
        # Triangle ABC → edges: A-B, B-C, C-A (rolled)
        a = elements.reshape(-1)
        b = torch.roll(elements, shifts=-1, dims=1).reshape(-1)

        # Build bidirectional edges
        src = torch.cat([a, b], dim=0)
        dst = torch.cat([b, a], dim=0)

        # Remove self-loops
        mask = src != dst
        src = src[mask]
        dst = dst[mask]

        # ----------------------------------------------------------
        # 2) Build sparse COO adjacency (values = True)
        # ----------------------------------------------------------
        idx = torch.stack([src, dst], dim=0)  # [2, nnz]

        # Values for COO; duplicates will be coalesced
        vals = torch.ones(idx.size(1), dtype=torch.bool, device="cuda")

        # Sparse adjacency (undirected)
        A = torch.sparse_coo_tensor(idx, vals, size=(n_points, n_points)).coalesce()

        # Now A.indices(): [2, nnz], A.indices()[0] = row_ids, A.indices()[1] = col_ids
        row = A.indices()[0]
        col = A.indices()[1]

        # ----------------------------------------------------------
        # 3) Convert COO → CSR row pointers
        # ----------------------------------------------------------
        # Count neighbors per row
        counts = torch.bincount(row, minlength=n_points)

        # CSR row pointer: rptr[i] = start index of row i
        rptr = torch.zeros(n_points + 1, dtype=torch.long, device="cuda")
        rptr[1:] = torch.cumsum(counts, dim=0)

        # ----------------------------------------------------------
        # 4) Fill the neighb array (vectorized)
        # ----------------------------------------------------------
        g.neighb = torch.full((n_points, max_neighbors + 1), -1, dtype=torch.int32, device="cuda")

        # Cap the neighbor count
        g.neighb[:, 0] = torch.clamp(counts, max=max_neighbors) - 1

        # Build an index grid for selecting up to max_neighbors neighbors
        # Example: for max_neighbors=15 → [0,1,2,...,14]
        k = torch.arange(max_neighbors, device="cuda")

        # For each node i, compute the global COO index range:
        # row i contains col[ rptr[i] : rptr[i+1] ]
        # We broadcast rptr against k to fetch all neighbors in one shot.
        row_start = rptr[:-1].unsqueeze(1)          # [n, 1]
        row_len   = counts.unsqueeze(1)             # [n, 1]

        # Mask: ensures we only select valid neighbors (< row_len)
        mask = (k < row_len)

        # COO indices to select
        sel = row_start + k                        # [n, max_neighbors]

        # Valid selections only
        sel = torch.where(mask, sel, torch.zeros_like(sel))

        # Gather neighbor indices
        neigh_vals = torch.where(mask, col[sel], torch.tensor(-1, device="cuda"))

        # Write neighbors into final array
        g.neighb[:, 1:] = neigh_vals.to(torch.int32)
        