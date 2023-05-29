using CSV
using DataFrames
using Plots
using Plots.PlotMeasures
using LaTeXStrings
using LinearAlgebra
using Rotations
using StaticArrays

# as: arrow head size 0-1 (fraction of arrow length)
# la: arrow alpha transparency 0-1
function arrow3d!(x, y, z,  u, v, w; as=0.3, lc=:black, la=0.1, lw=1, scale=:identity)
    (as < 0) && (nv0 = -maximum(norm.(eachrow([u v w]))))
    for (x,y,z, u,v,w) in zip(x,y,z, u,v,w)
        nv = sqrt(u^2 + v^2 + w^2)
        v1, v2 = -[u,v,w]/nv, nullspace(adjoint([u,v,w]))[:,1]
        v4 = (3*v1 + v2)/3.1623  # sqrt(10) to get unit vector
        v5 = v4 - 2*(v4'*v2)*v2
        (as < 0) && (nv = nv0) 
        v4, v5 = -as*nv*v4, -as*nv*v5
        plot!([x,x+u], [y,y+v], [z,z+w], lc=lc, la=la, lw=lw, scale=scale, label=false)
        plot!([x+u,x+u-v5[1]], [y+v,y+v-v5[2]], [z+w,z+w-v5[3]], lc=lc, la=la, lw=lw, label=false)
        plot!([x+u,x+u-v4[1]], [y+v,y+v-v4[2]], [z+w,z+w-v4[3]], lc=lc, la=la, lw=lw, label=false)
    end
end

function plot_data(name, path)
    df = DataFrame(CSV.File(path, comment = "#"));
    
    p1 = plot(df.t, df.q1, label = L"q_1", ylabel = L"|q|", legend = :topright, xaxis = false, bottom_margin = -5mm, dpi=800)
    p1 = plot!(df.t, df.q2, label = L"q_2")
    p1 = plot!(df.t, df.q3, label = L"q_3")
    p1 = plot!(df.t, df.q0, label = L"q_0")

    p2 = plot(df.t, df.w1, label = L"\omega_1", ylabel = L"\omega (rad/s)", legend = :topright, xaxis = false, bottom_margin = -5mm)
    # p2 = plot(df.t, df.w1, label = L"\omega_1", ylabel = L"\omega (rad/s)", xlabel = L"Time (s)", legend = :topright)
    p2 = plot!(df.t, df.w2, label = L"\omega_2")
    p2 = plot!(df.t, df.w3, label = L"\omega_3")
   
    p3 = plot(df.t, df.u1, label = L"u_1", xlabel = L"Time (s)", ylabel = L"u (Nm)", legend = :topright)
    p3 = plot!(df.t, df.u2, label = L"u_2")
    p3 = plot!(df.t, df.u3, label = L"u_3")

    plot(p1, p2, p3, size = (1000.0, 1000.0), layout = grid(3, 1, heights = (0.5, 0.25, 0.25)))
    # plot(p1, p2, size = (1000.0, 1000.0), layout = grid(2, 1, heights = (0.66, 0.33)))
    savefig("images/" * name * "_attiude.png")

    p1 = plot(df.t, df.q1_b_lvlh, label = L"q_1", ylabel = L"|q|", legend = :topright, xaxis = false, bottom_margin = -5mm, dpi=800)
    p1 = plot!(df.t, df.q2_b_lvlh, label = L"q_2")
    p1 = plot!(df.t, df.q3_b_lvlh, label = L"q_3")
    p1 = plot!(df.t, df.q0_b_lvlh, label = L"q_0")

    p2 = plot(df.t, df.w1_b_lvlh, label = L"\omega_1", ylabel = L"\omega (rad/s)", legend = :topright, xaxis = false, bottom_margin = -5mm)
    # p2 = plot(df.t, df.w1_b_lvlh, label = L"\omega_1", ylabel = L"\omega (rad/s)", xlabel = L"Time (s)", legend = :topright)
    p2 = plot!(df.t, df.w2_b_lvlh, label = L"\omega_2")
    p2 = plot!(df.t, df.w3_b_lvlh, label = L"\omega_3")
   
    p3 = plot(df.t, df.u1, label = L"u_1", xlabel = L"Time (s)", ylabel = L"u (Nm)", legend = :topright)
    p3 = plot!(df.t, df.u2, label = L"u_2")
    p3 = plot!(df.t, df.u3, label = L"u_3")

    plot(p1, p2, p3, size = (1000.0, 1000.0), layout = grid(3, 1, heights = (0.5, 0.25, 0.25)))
    # plot(p1, p2, size = (1000.0, 1000.0), layout = grid(2, 1, heights = (0.66, 0.33)))
    savefig("images/" * name * "_attiude_lvlh.png")


    t_o = 5100
    r_o = 1000e3
    n_o = 1

    for i in 1:n_o
        t_min = (i - 1) * t_o
        t_max = i * t_o
        df_orbit = filter(row -> (row.t >= t_min && row.t < t_max), df)
        plot(df_orbit.x / r_o, df_orbit.y / r_o, df_orbit.z / r_o, formatter=(_...) -> "", legend = :none, dpi=800)

        df_orbit = filter!(row-> (row.t % 200 == 0), df_orbit)
        transform!(df_orbit, [:q0, :q1, :q2, :q3] => ByRow((q0, q1, q2, q3) -> (QuatRotation(q0, q1, q2, q3) * SVector(1.0, 0.0, 0.0))) => [:x_body_x, :x_body_y, :x_body_z])
        transform!(df_orbit, [:q0, :q1, :q2, :q3] => ByRow((q0, q1, q2, q3) -> (QuatRotation(q0, q1, q2, q3) * SVector(0.0, 1.0, 0.0))) => [:y_body_x, :y_body_y, :y_body_z])
        transform!(df_orbit, [:q0, :q1, :q2, :q3] => ByRow((q0, q1, q2, q3) -> (QuatRotation(q0, q1, q2, q3) * SVector(0.0, 0.0, 1.0))) => [:z_body_x, :z_body_y, :z_body_z])
        # transform!(df_orbit, [:q0_lvlh, :q1_lvlh, :q2_lvlh, :q3_lvlh] => ByRow((q0, q1, q2, q3) -> (QuatRotation(q0, q1, q2, q3) * SVector(1.0, 0.0, 0.0))) => [:x_body_x, :x_body_y, :x_body_z])
        # transform!(df_orbit, [:q0_lvlh, :q1_lvlh, :q2_lvlh, :q3_lvlh] => ByRow((q0, q1, q2, q3) -> (QuatRotation(q0, q1, q2, q3) * SVector(0.0, 1.0, 0.0))) => [:y_body_x, :y_body_y, :y_body_z])
        # transform!(df_orbit, [:q0_lvlh, :q1_lvlh, :q2_lvlh, :q3_lvlh] => ByRow((q0, q1, q2, q3) -> (QuatRotation(q0, q1, q2, q3) * SVector(0.0, 0.0, 1.0))) => [:z_body_x, :z_body_y, :z_body_z])
        
        scatter!(df_orbit.x / r_o, df_orbit.y / r_o, df_orbit.z / r_o, markersize = 0.2)
        arrow3d!(df_orbit.x / r_o, df_orbit.y / r_o, df_orbit.z / r_o, df_orbit.x_body_x, df_orbit.x_body_y, df_orbit.x_body_z; as=-0.01, lc=:red, la=0.5)
        arrow3d!(df_orbit.x / r_o, df_orbit.y / r_o, df_orbit.z / r_o, df_orbit.y_body_x, df_orbit.y_body_y, df_orbit.y_body_z; as=-0.01, lc=:green, la=0.5)
        arrow3d!(df_orbit.x / r_o, df_orbit.y / r_o, df_orbit.z / r_o, df_orbit.z_body_x, df_orbit.z_body_y, df_orbit.z_body_z; as=-0.01, lc=:blue, la=0.5)
        savefig("images/" * name * "_orbit_" * string(t_min) * "s_to_" * string(t_max) * "s.png")
    end
end

name = "pd_noise"
path = "out.csv"

plot_data(name, path)
