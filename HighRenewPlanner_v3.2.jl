using JuMP, PowerModels, Gurobi, Plots,  CSV, DataFrames, LinearAlgebra, TickTock, Clustering

# Total simulation time steps 
num_time_sim = 12 # total: 8784
dt = 2  # hour resolution
time_ini = 5375 - 17 # 5375-17 #  
T_sim = time_ini:dt:(time_ini + dt*num_time_sim - 1)

# ------------------ Buses ---------------------
bus_data = DataFrame(CSV.File("ERCOTSystem\\texas_bus.csv"))
num_bus = size(bus_data, 1)
bus_data.rowindex = 1:num_bus  # add a row order index
print("Total bus number is ", num_bus, "\n")

# ------------------ Loads ------------------
# (net load of nodal hydro generation)
load_data = DataFrame(CSV.File("ERCOTSystem\\loads.csv"))
timeslots = load_data[!,"UTC Time"]
num_time = size(load_data,1)
num_load = size(load_data,2) - 1
print("Total time number is ", num_time, "\n")

# 1 - direct from load csv that arrange load in the order of bus
p_load_t_all = Matrix(load_data[:,2:end])'/100    # unit: MW
p_load_t = p_load_t_all[:, T_sim]

p_load_max_t = abs.(p_load_t);
p_load_min_t = 0.5*p_load_max_t;

# total load over time
total_load_t = sum(p_load_t_all, dims=1)

# Load Peak Time
max_L_index = findfirst(x -> x == maximum(total_load_t), total_load_t)[2]
print("Load Peak: ", maximum(total_load_t), "(MW) at ", timeslots[max_L_index], "\n", "Time Index: ", max_L_index, "\n")
#plot((5375-17-24*10:5375+7+24*10),total_load_t[5375-17-24*10:5375+7+24*10], xlabel = "Time (h)", ylabel = "Loads (MW)", title = "Total Load Trajectory Over One Year")

# Load Shedding 
ratio_shed_t = 0.2 
ratio_shed_all = 0.2
L_shed_max_t = ratio_shed_t*total_load_t
L_shed_max = ratio_shed_all*sum(total_load_t)

#plot load trajectories
# L1 = plot(total_load_t', xlabel = "Time (h)", ylabel = "Loads (MW)", title = "Total Load Trajectory Over One Year")
# display(L1)
# ------------------ Branch ------------------
branch_data = DataFrame(CSV.File("ERCOTSystem\\texas_branch.csv"))
num_branch = size(branch_data, 1)
print("Total branch number is ", num_branch)

# construct key branch matrices 
A_nb = zeros(num_bus, num_branch); #node-branch incidence matrix
X_bn = zeros(num_branch, num_bus)  #branch-node resistance matrix
P_br_max = zeros(num_branch, num_time_sim);  # branch transmission capacity
for i = 1:num_branch
    P_br_max[i,:] = (branch_data[i,:].rateA + branch_data[i,:].rateB + branch_data[i,:].rateC)*ones(1,num_time_sim);
    
    f_bus_id = branch_data[i,:].from_bus_id
    t_bus_id = branch_data[i,:].to_bus_id
    f_bus = bus_data[bus_data.bus_id .== f_bus_id, :].rowindex
    t_bus = bus_data[bus_data.bus_id .== t_bus_id, :].rowindex
    
    if length(f_bus)>=2 || length(t_bus)>=2
        print("Error in bus selection!") # in case of getting multiple buses
    end
    
    x_br  = branch_data[i,:].x
    
    X_bn[i, f_bus] .=  1/x_br
    X_bn[i, t_bus] .= -1/x_br
    A_nb[f_bus, i] .=  1 
    A_nb[t_bus, i] .= -1 
end

P_br_min = - P_br_max;

# nodal solar/wind max installation capacity
cap_sw_data = DataFrame(CSV.File("ERCOTSystem\\CPA_to_nodes.csv"))
cap_sw_data = coalesce.(cap_sw_data, 0.0)

# ------------------ Solar ------------------
solar_data = DataFrame(CSV.File("ERCOTSystem\\solars.csv"))
coe_PV_t_all = Matrix(solar_data[:,2:end])'/100    # unit: MW
coe_PV_t = coe_PV_t_all[:,T_sim]
# nodal total yearly solar coefficient 
coe_PV_all = sum(coe_PV_t_all, dims=2)

# plot solar power trajectories
# f1pv = plot(sum(coe_PV_t_all, dims=1)',  xlabel = "Time (h)", ylabel = "PV Gen. Factor") #seriestype = :scatter,
# f2pv = plot(coe_PV_all, xlabel = "Node Index", ylabel = "Yearly PV Energy (MWh) per 1MW")
# display(plot(f1pv, f2pv, layout = (2, 1), legend = false))

cap_PV = cap_sw_data[:,2] # unit: MW

# ------------------ Wind ------------------
wind_data = DataFrame(CSV.File("ERCOTSystem\\winds.csv"))
coe_WD_t_all = Matrix(wind_data[:,2:end])'/100    # unit: MW
coe_WD_t = coe_WD_t_all[:,T_sim]
# nodal total yearly wind coefficient
coe_WD_all = sum(coe_WD_t_all, dims=2)

# plot wind power trajectories
# f1wd = plot(sum(coe_WD_t_all, dims = 1)', xlabel = "Time (h)", ylabel = "Wind Gen. Factor")# seriestype = :scatter,
# f2wd = plot(coe_WD_all, xlabel = "Node Index", ylabel = "Yearly Wind Energy (MWh) per 1MW")
# display(plot(f1wd, f2wd, layout = (2, 1), legend = false))

cap_WD = cap_sw_data[:,3];

# ============= Select K typical days by K-means clustering ===========


# convert data by days
num_day = Int(num_time/24) -1 # 365 days

coe_WD_day_all = zeros(num_bus*24,num_day)
coe_PV_day_all = zeros(num_bus*24,num_day)
p_load_day_all = zeros(num_bus*24,num_day)

hour_lag = 5  # initial time may not be 0 AM, add a lag to handle
for i = 1:num_day
   coe_WD_day_all[:,i] = vec(coe_WD_t_all[:,24*i-23+hour_lag:24*i+hour_lag]);
   coe_PV_day_all[:,i] = vec(coe_PV_t_all[:,24*i-23+hour_lag:24*i+hour_lag]);
   p_load_day_all[:,i] = vec(p_load_t_all[:,24*i-23+hour_lag:24*i+hour_lag]);  
end

#p_load_day_all_nor = abs.(p_load_day_all)./(maximum(abs.(p_load_day_all), dims = 2).+1e-8)

feature_day = vcat(p_load_day_all, coe_WD_day_all, coe_PV_day_all)

K_day = 9 # number of clusters

cluster_days = kmeans(feature_day, K_day; maxiter=500, display=:iter)
typical_days = cluster_days.centers; # get the cluster centers
Assign_day = assignments(cluster_days) # get the assignments of points to clusters
Count_day = counts(cluster_days) # number of points in each cluster
print("Number of days in clusters: ", Count_day,"\n")

# Load curves in typical days
load_typical = sum(reshape(typical_days[1:num_bus*24,1],(num_bus,24) ),dims=1)
# fig_ty = plot(load_typical', xlabel = "Hours (h)",markershape =:circle, ylabel = "Load (MW)", title = "Typical Load Curves")
print("Peak Load is ", round(maximum(load_typical),digits=3), " (MW)\n")
for i = 2:K_day
    load_typical = sum(reshape(typical_days[1:num_bus*24,i],(num_bus,24)),dims=1)
    print("Peak Load is ", round(maximum(load_typical),digits=3), " (MW)\n")
    # fig_ty = plot!(load_typical', xlabel = "Hours (h)", markershape =:circle, ylabel = "Load (MW)", title = "Typical Load Curves") 
end
# display(fig_ty)

# wind curve
wd_typical = sum(reshape(typical_days[num_bus*24+1:2*num_bus*24,1],(num_bus,24)),dims=1)
# fig_ty = plot(wd_typical', xlabel = "Hours (h)",markershape =:circle, ylabel = "Wind Coefficient", title = "Typical Wind Curves") 
for i = 2:K_day
    wd_typical = sum(reshape(typical_days[num_bus*24+1:2*num_bus*24,i],(num_bus,24)),dims=1)
    # fig_ty = plot!(wd_typical', xlabel = "Hours (h)", markershape =:circle, ylabel = "Wind Coefficient", title = "Typical Wind Curves") 
end
# display(fig_ty)

#Solar curve
pv_typical = sum(reshape(typical_days[2*num_bus*24+1:3*num_bus*24,1],(num_bus,24)),dims=1)
# fig_ty = plot(pv_typical', xlabel = "Hours (h)",markershape =:circle, ylabel = "Solar Coefficient", title = "Typical Solar Curves") 
for i = 2:K_day
    pv_typical = sum(reshape(typical_days[2*num_bus*24+1:3*num_bus*24,i],(num_bus,24)),dims=1)
    # fig_ty = plot!(pv_typical', xlabel = "Hours (h)", markershape =:circle, ylabel = "Solar Coefficient", title = "Typical Solar Curves") 
end
# display(fig_ty)

# Define Simulation Scenarios
num_scenario = K_day + 1

coe_PV_t_sim = zeros(num_bus, num_time_sim, num_scenario)
coe_WD_t_sim = zeros(num_bus, num_time_sim, num_scenario)
p_load_t_sim = zeros(num_bus, num_time_sim, num_scenario)

# extreme day
p_load_t_sim[:,:,1] = p_load_max_t
coe_WD_t_sim[:,:,1] = coe_WD_t
coe_PV_t_sim[:,:,1] = coe_PV_t;

for s = 2:num_scenario
    p_load_t_sim[:,:,s] = reshape(typical_days[1:num_bus*24,s-1],(num_bus,24))[:,1:dt:24]
    coe_WD_t_sim[:,:,s] = reshape(typical_days[num_bus*24+1:2*num_bus*24,s-1],(num_bus),24)[:,1:dt:24]
    coe_PV_t_sim[:,:,s] = reshape(typical_days[2*num_bus*24+1:3*num_bus*24,s-1],(num_bus),24)[:,1:dt:24]
end

days_sce = vcat(1,Count_day);

# t_plot = (0:2:22)

# fsim1 = plot(t_plot, sum(p_load_t_sim[:,:,1], dims = 1)'/1e3, label = "Peak Load", xlabel = "Hours (h)",markershape =:circle, ylabel = "Load (GW)" )
# for s = 2:num_scenario
#   fsim1 = plot!(t_plot, sum(p_load_t_sim[:,:,s], dims = 1)'/1e3,xlabel = "Hours (h)",markershape =:circle, ylabel = "Load (GW)")
# end
# display(fsim1)

# fsim2 = plot(t_plot, sum(coe_WD_t_sim[:,:,1], dims = 1)'/2e3, label = "Peak Load", xlabel = "Hours (h)",markershape =:circle, ylabel = "Average Wind Generation Factor" )
# for s = 2:num_scenario
#   fsim2 = plot!(t_plot, sum(coe_WD_t_sim[:,:,s], dims = 1)'/2e3,xlabel = "Hours (h)",markershape =:circle, ylabel = "Average Wind Generation Factor" )
# end
# display(fsim2)

# fsim3 = plot(t_plot, sum(coe_PV_t_sim[:,:,1], dims = 1)'/2e3, label = "Peak Load", xlabel = "Hours (h)",markershape =:circle, ylabel = "Average Solar Generation Factor" )
# for s = 2:num_scenario
#   fsim3 = plot!(t_plot, sum(coe_PV_t_sim[:,:,s], dims = 1)'/2e3,legend=:topleft, xlabel = "Hours (h)",markershape =:circle, ylabel = "Average Solar Generation Factor" )
# end
# display(fsim3)


# ------------------ All Existing Power Plants ------------------
plant_data = DataFrame(CSV.File("ERCOTSystem\\texas_plant.csv"))
num_plant = size(plant_data, 1)
plant_data.rowindex = 1:num_plant  # add a row order index
print("Total power plants number is ", num_plant, "\n")

print("Power plant types include ", unique(plant_data.type),"\n")
plant_data_solar = plant_data[ plant_data.type .== "solar", :]
plant_data_wind = plant_data[ plant_data.type .== "wind", :]
plant_data_ng = plant_data[ plant_data.type .== "ng", :] 
plant_data_coal = plant_data[ plant_data.type .== "coal", :]
plant_data_hydro = plant_data[ plant_data.type .== "hydro", :] 
plant_data_nuclear = plant_data[ plant_data.type .== "nuclear", :] 
                            
pg_solar = sum(plant_data_solar.Pmax)
pg_wind = sum(plant_data_wind.Pmax)
pg_ng = sum(plant_data_ng.Pmax)
pg_coal = sum(plant_data_coal.Pmax)
pg_hydro = sum(plant_data_hydro.Pmax)
pg_nuclear = sum(plant_data_nuclear.Pmax)
pg_sum = sum(plant_data.Pmax)

labels = ["solar"; "wind"; "gas";"coal"; "hydro"; "nuclear"] 
sizes = [pg_solar; pg_wind; pg_ng; pg_coal; pg_hydro; pg_nuclear]
print("solar: ", round(pg_solar, digits =3), " MW (", round(pg_solar/pg_sum*100, digits =3), "%) \n", 
      "wind: ", round(pg_wind, digits =3), " MW (", round(pg_wind/pg_sum*100, digits =3), "%) \n", 
      "natural gas: ", round(pg_ng, digits =3), " MW (", round(pg_ng/pg_sum*100, digits =3), "%) \n", 
      "coal: ",  round(pg_coal, digits =3), " MW (", round(pg_coal/pg_sum*100, digits =3), "%) \n", 
      "hydro: ",  round(pg_hydro, digits =3), " MW (", round(pg_hydro/pg_sum*100, digits =3), "%) \n", 
      "nuclear: ",  round(pg_nuclear, digits =3), " MW (", round(pg_nuclear/pg_sum*100, digits =3), "%) \n", 
      "clean energy: ",  round(pg_nuclear+pg_solar+pg_wind+pg_hydro, digits =3), " MW (", round((pg_nuclear+pg_solar+pg_wind+pg_hydro)/pg_sum*100, digits =3), "%) \n", 
      "total: ", round(pg_sum, digits =3), " MW \n")

#plot(labels, sizes,seriestype = :pie )

# ----------------- Parameters for Existing Power Plants -----------
num_gen_ng      = size(plant_data_ng,1)
num_gen_coal    = size(plant_data_coal,1)
num_gen_solar   = size(plant_data_solar,1)
num_gen_wind    = size(plant_data_wind,1)
num_gen_hydro   = size(plant_data_hydro,1)
num_gen_nuclear = size(plant_data_nuclear,1)

# generation capacity over time
Pmin_ng = plant_data_ng.Pmin.*ones(1,num_time_sim)
Pmax_ng = plant_data_ng.Pmax.*ones(1,num_time_sim)
ramp_ng_max = dt*2*plant_data_ng.ramp_30
ramp_ng_min = - ramp_ng_max 

Pmin_coal = plant_data_coal.Pmin.*ones(1,num_time_sim)
Pmax_coal = plant_data_coal.Pmax.*ones(1,num_time_sim)
ramp_coal_max = dt*2*plant_data_coal.ramp_30
ramp_coal_min = - ramp_coal_max 

# Construct nodal matrices for power flow
A_coal = zeros(num_bus,num_gen_coal)
A_ng = zeros(num_bus,num_gen_ng)
A_wind = zeros(num_bus,num_gen_wind)
A_solar = zeros(num_bus,num_gen_solar)
A_nuclear = zeros(num_bus,num_gen_nuclear)

for i = 1:num_gen_coal
    bus_index = bus_data[bus_data.bus_id .== plant_data_coal[i,:].bus_id, :].rowindex[1]
    A_coal[bus_index, i] = 1
end

for i = 1:num_gen_ng
    bus_index = bus_data[bus_data.bus_id .== plant_data_ng[i,:].bus_id, :].rowindex[1]
    A_ng[bus_index, i] = 1
end

for i = 1:num_gen_nuclear
    bus_index = bus_data[bus_data.bus_id .== plant_data_nuclear[i,:].bus_id, :].rowindex[1]
    A_nuclear[bus_index, i] = 1
end
P_nuclear = plant_data_nuclear.Pg.*ones(1,num_time_sim)

solar_index = Int32.(zeros(num_gen_solar,1))
for i =1: num_gen_solar
    bus_index = bus_data[bus_data.bus_id .== plant_data_solar[i,:].bus_id, :].rowindex[1]
    A_solar[bus_index, i] = 1
    solar_index[i] = bus_index
end
Pmax_solar = plant_data_solar.Pmax.*coe_PV_t_sim[solar_index[:,1], :,:]

wind_index = Int32.(zeros(num_gen_wind,1))
for i =1: num_gen_wind
    bus_index = bus_data[bus_data.bus_id .== plant_data_wind[i,:].bus_id, :].rowindex[1]
    A_wind[bus_index, i] = 1
    wind_index[i] = bus_index 
end
Pmax_wind = plant_data_wind.Pmax.*coe_WD_t_sim[wind_index[:,1], :,:];

# Energy Storage 
cap_ES_max = 2000 # MWh
cap_ES = cap_ES_max*ones(num_bus,1)[:,1]
#cap_ES[800:end] .= 0
 
p_dis_max = 0.25 # MW
p_cha_max = 0.25 # MW
e_max = 1  #MWh
ES_ini = 0.5  # initial SOC is 50%
kap_ES = 0.99 # 100% no leakage

alp_cha = sqrt(0.85)
alp_dis = sqrt(0.85)

# Cost parameters
gencost_data = DataFrame(CSV.File("ERCOTSystem\\texas_gencost.csv"))

c_gen_ng = hcat(gencost_data[plant_data_ng.rowindex,:].c2, 
                gencost_data[plant_data_ng.rowindex,:].c1, 
                gencost_data[plant_data_ng.rowindex,:].c0)/1e3 

c_gen_coal = hcat(gencost_data[plant_data_coal.rowindex,:].c2, 
                  gencost_data[plant_data_coal.rowindex,:].c1, 
                  gencost_data[plant_data_coal.rowindex,:].c0)/1e3  

# investment
c_PV_invest = 890 # k$/MW
c_WD_invest = 1212 # k$/MW
c_ES_invest = 369 # k$/MWh
c_curtail = 0.002 # k$/MWh
c_ES_oper = 0.001  # k$/MWh
c_loadshed = 5  # k$/MWh
c_coal_retire = 117 # k$/MW
c_ng_retire = 15 # k$/MW

# Existing Nodal Capacity of Generation Disrribution
nc_coal = A_coal*Pmax_coal[:,1]
nc_ng = A_ng*Pmax_ng[:,1]
nc_wind = A_wind*plant_data_wind.Pmax
nc_solar = A_solar*plant_data_solar.Pmax;

# plot(nc_coal, label = "coal")
# plot!(nc_ng, label= "gas")
# plot!(nc_wind, label= "wind")
# plot!(nc_solar, label= "solar", xlabel = "Node Index", ylabel = "Gen. Capacity (MW)", linecolor = "black", title = "Existing Plants Map")


# ----------------- Line Capacity Expansion
lineexp_data = DataFrame(CSV.File("ERCOTSystem\\line_upgrade_costs.csv"))
cost_lineexp = lineexp_data.cost # k$/MW
cap_lineexp = 2000*ones(num_branch,1);  # maximum capacity increase 2000 MW



# ============== Optimization Model ========================
## ================= Select Optimizer =====================
model_plan = Model(Gurobi.Optimizer)

num_scenario = 1

## ================= Create Variables and Constraints =====================
# Solar  
@variable(model_plan, 0 <= w_PV_var[1:num_bus])
@variable(model_plan, 0 <= p_PV_var[1:num_bus, 1:num_time_sim, 1:num_scenario])
@constraint(model_plan, w_PV_var .<= cap_PV)

# Wind  
@variable(model_plan, 0 <= w_WD_var[1:num_bus])
@variable(model_plan, 0 <= p_WD_var[1:num_bus, 1:num_time_sim, 1:num_scenario])
@constraint(model_plan, w_WD_var .<= cap_WD)


# Existing Solar and Wind
@variable(model_plan, 0 <= p_solar_var[1:num_gen_solar, 1:num_time_sim, 1:num_scenario])
@variable(model_plan, 0 <= p_wind_var[1:num_gen_wind, 1:num_time_sim, 1:num_scenario])

for s=1:num_scenario
   @constraint(model_plan, p_PV_var[:,:,s] .<= w_PV_var.*coe_PV_t_sim[:,:,s])
   @constraint(model_plan, p_WD_var[:,:,s] .<= w_WD_var.*coe_WD_t_sim[:,:,s])
   @constraint(model_plan, p_wind_var[:,:,s] .<= Pmax_wind[:,:,s])
   @constraint(model_plan, p_solar_var[:,:,s] .<= Pmax_solar[:,:,s])
end


# Coal Generators
#@variable(model_plan, z_coal_var[1:num_gen_coal], Bin) # coal
@variable(model_plan, 0 <= z_coal_var[1:num_gen_coal] <= 1) # coal
@variable(model_plan, p_coal_var[1:num_gen_coal, 1:num_time_sim, 1:num_scenario])

for s = 1:num_scenario
   @constraint(model_plan, z_coal_var.*Pmin_coal .<= p_coal_var[:,:,s])
   @constraint(model_plan, p_coal_var[:,:,s] .<= z_coal_var.*Pmax_coal)
   for t = 2:num_time_sim
      @constraint(model_plan,  ramp_coal_min .<= p_coal_var[:,t,s] - p_coal_var[:,t-1,s] .<= ramp_coal_max )
   end
end

# Natural Gass
#@variable(model_plan, z_ng_var[1:num_gen_ng], Bin) # natural gas
@variable(model_plan, 0 <= z_ng_var[1:num_gen_ng] <= 1) # natural gas
@variable(model_plan, p_ng_var[1:num_gen_ng, 1:num_time_sim, 1:num_scenario])

for s = 1:num_scenario
   @constraint(model_plan, z_ng_var.*Pmin_ng .<= p_ng_var[:,:,s])
   @constraint(model_plan, p_ng_var[:,:,s] .<= z_ng_var.*Pmax_ng);
   for t = 2:num_time_sim
      @constraint(model_plan,  ramp_ng_min .<= p_ng_var[:,t,s] - p_ng_var[:,t-1,s] .<= ramp_ng_max )
   end
end
# Adjustable Loads
#@variable(model_plan, 0 <= p_load_var[1:num_load, 1:num_time_sim])
#@constraint(model_plan, p_load_min_t .<= p_load_var .<= p_load_max_t)

##  Energy Storage
alp_cha = 1
alp_dis = 1

# --------------------------- full model ------------------------
# @variable(model_plan, nu_ES_var[1:num_bus, 1:num_time_sim], Bin)
@variable(model_plan, 0 <= nu_ES_var[1:num_bus, 1:num_time_sim,1:num_scenario] <= 1)
@variable(model_plan, 0 <= w_ES_var[1:num_bus])
@variable(model_plan, 0 <= y_ES_var[1:num_bus, 1:num_time_sim,1:num_scenario] )
@variable(model_plan, 0 <= p_dis_var[1:num_bus, 1:num_time_sim,1:num_scenario])
@variable(model_plan, 0 <= p_cha_var[1:num_bus, 1:num_time_sim,1:num_scenario])
@variable(model_plan, 0 <= e_ES_var[1:num_bus, 1:num_time_sim,1:num_scenario])

big_M = 1e3
@constraint(model_plan, w_ES_var .<= cap_ES)

for s = 1:num_scenario
   @constraint(model_plan, p_dis_var[:,:,s] .<= p_dis_max*y_ES_var[:,:,s] )
   @constraint(model_plan, p_cha_var[:,:,s] .<= p_cha_max*( w_ES_var*ones(1,num_time_sim) - y_ES_var[:,:,s]))
   @constraint(model_plan, y_ES_var[:,:,s] .<= w_ES_var*ones(1,num_time_sim)) 
   @constraint(model_plan, y_ES_var[:,:,s] .<= big_M*nu_ES_var[:,:,s])

   for t = 1:num_time_sim
       @constraint(model_plan, big_M*nu_ES_var[:,t,s] - big_M*ones(num_bus,1) + w_ES_var .<= y_ES_var[:,t,s])
       if t == 1
           @constraint(model_plan, e_ES_var[:,t,s] .== kap_ES*(ES_ini*e_max*w_ES_var) + dt*(alp_cha*p_cha_var[:,t,s] - 1/alp_dis*p_dis_var[:,t,s]))
       else
           @constraint(model_plan, e_ES_var[:,t,s] .== kap_ES*e_ES_var[:,t-1,s] + dt*(alp_cha*p_cha_var[:,t,s] - 1/alp_dis*p_dis_var[:,t,s]))
       end
       @constraint(model_plan, e_ES_var[:,t,s] .<= e_max*w_ES_var)
   end
   @constraint(model_plan, e_ES_var[:,num_time_sim,s] .== ES_ini*e_max*w_ES_var); # recover to the initial SOC
end

# # DC Power Flow
@variable(model_plan, P_br_var[1:num_branch, 1:num_time_sim,1:num_scenario])
@variable(model_plan, tha_var[1:(num_bus-1), 1:num_time_sim,1:num_scenario])

# @variable(model_plan, 0 <= cap_br_add_var[1:num_branch]);
# @constraint(model_plan, cap_br_add_var .<= cap_lineexp );

# Power Flow
for s = 1:num_scenario
   @constraint(model_plan, P_br_min .<= P_br_var[:,:,s] .<= P_br_max)
   # Consider line expansion
   # @constraint(model_plan, P_br_var[:,:,s] .<= P_br_max + cap_br_add_var*ones(1,num_time_sim))
   # @constraint(model_plan, P_br_min .<= P_br_var[:,:,s] + cap_br_add_var*ones(1,num_time_sim))


   @constraint(model_plan, P_br_var[:,:,s] .== X_bn[:,2:end]*tha_var[:,:,s])
   @constraint(model_plan, A_nb*P_br_var[:,:,s] .== p_PV_var[:,:,s] + p_WD_var[:,:,s] + p_dis_var[:,:,s] - p_cha_var[:,:,s]
                 - p_load_t_sim[:,:,s] + A_coal*p_coal_var[:,:,s] + A_ng*p_ng_var[:,:,s] 
                 + A_wind*p_wind_var[:,:,s] + A_solar*p_solar_var[:,:,s] + A_nuclear*P_nuclear);
end
# Renewable Penetration
renew_pene = 0.8
for s = 1:num_scenario
   @constraint(model_plan, sum(p_coal_var[:,:,s]) + sum(p_ng_var[:,:,s]) <= (1 - renew_pene)*sum(p_load_t_sim[:,:,s]));
end
#@constraint(model_plan, sum(w_PV_var + w_WD_var) + sum() >= gam_C*sum(z_gen_var.*cap_gen) )


## ================= Define Objective =====================
num_years = 10*365 #10

# ==== one-shot planning cost
Obj_invest = sum(c_PV_invest*w_PV_var) +  sum(c_WD_invest*w_WD_var) + sum(c_ES_invest*w_ES_var)
Obj_retire_coal =  sum(c_coal_retire*(ones(num_gen_coal,1)-z_coal_var).*Pmax_coal[:,1]) 
Obj_retire_ng =  sum(c_ng_retire*(ones(num_gen_ng,1)-z_ng_var).*Pmax_ng[:,1]) 
# Obj_line_expansion =   sum(cost_lineexp.*cap_br_add_var)

# ==== operation cost
Obj_op_solar = zeros(num_scenario,1)*w_PV_var[1] # just make the optimization data format 
Obj_op_wind  = zeros(num_scenario,1)*w_WD_var[1]
Obj_op_ES  = zeros(num_scenario,1)*w_WD_var[1]
Obj_op_coal  = zeros(num_scenario,1)*w_WD_var[1]
Obj_op_ng  = zeros(num_scenario,1)*w_WD_var[1]



for s = 1:num_scenario
    Obj_op_solar[s] = dt*c_curtail*(sum(w_PV_var.*coe_PV_t_sim[:,:,s]-p_PV_var[:,:,s])+ sum(Pmax_solar[:,:,s]-p_solar_var[:,:,s]))
    
    Obj_op_wind[s]  = dt*c_curtail*(sum(w_WD_var.*coe_WD_t_sim[:,:,s]-p_WD_var[:,:,s])+ sum(Pmax_wind[:,:,s] - p_wind_var[:,:,s]))
    
    Obj_op_ES[s] = dt*c_ES_oper*sum(p_dis_var[:,:,s] + p_cha_var[:,:,s])
        
    for i = 1:num_gen_coal
       Obj_op_coal[s] =  Obj_op_coal[s] + dt*sum(c_gen_coal[i,2]*p_coal_var[i,:,s] + z_coal_var[i]*c_gen_coal[i,3]*ones(num_time_sim,1))
                       + dt*sum(c_gen_coal[i,1]*(p_coal_var[i,:,s].^2) )
    end

    for i = 1:num_gen_ng
       Obj_op_ng[s] =  Obj_op_ng[s] + dt*sum(c_gen_ng[i,2]*p_ng_var[i,:,s] + z_ng_var[i]*c_gen_ng[i,3]*ones(num_time_sim,1))
                       + dt*sum(c_gen_ng[i,1]*(p_ng_var[i,:,s].^2))
    end
    # Obj_opera_year = Obj_opera_year + days_sce[s]*(Obj_op_solar[s] + Obj_op_wind[s] + Obj_op_ES[s]
    #                                                                + Obj_op_coal[s] + Obj_op_ng[s] )
end

Obj_opera_year = sum( days_sce[s]*(Obj_op_solar[s] + Obj_op_wind[s] + Obj_op_ES[s]
                      + Obj_op_coal[s] + Obj_op_ng[s] )  for s = 1:num_scenario )

Obj_exp = Obj_invest + Obj_retire_coal + Obj_retire_ng + num_years*Obj_opera_year  # + Obj_line_expansion

@objective(model_plan, Min, Obj_exp);

## ================= Solve model =======================
tick()
optimize!(model_plan)
tock()


# Get and Save optimal solution values
w_PV = value.(w_PV_var)
p_PV = value.(p_PV_var)
w_WD = value.(w_WD_var)
p_WD = value.(p_WD_var)
p_solar = value.(p_solar_var)
p_wind = value.(p_wind_var)
z_coal = value.(z_coal_var)
p_coal = value.(p_coal_var)
z_ng = value.(z_ng_var)
p_ng = value.(p_ng_var)
nu_ES = value.(nu_ES_var)
w_ES = value.(w_ES_var)
y_ES = value.(y_ES_var)
p_dis = value.(p_dis_var)
p_cha = value.(p_cha_var)
e_ES = value.(e_ES_var)
P_br = value.(P_br_var)
tha = value.(tha_var)
P_br_ava = P_br_max - abs.(P_br)  # ==0: binding line
p_load = p_load_max_t

## ================= Show Results =======================
E_solar = sum(p_PV) + sum(p_solar)
E_wind = sum(p_WD) + sum(p_wind)
E_coal = sum(p_coal)
E_ng = sum(p_ng)
E_nuclear = sum(P_nuclear)
E_gen = E_solar + E_wind + E_coal + E_ng + E_nuclear
E_load = sum(p_load)


print("solar: ", round(E_solar, digits =3), " MWh (", round(E_solar/E_gen*100, digits =3), "%) \n", 
      "wind: ", round(E_wind, digits =3), " MWh (", round(E_wind/E_gen*100, digits =3), "%) \n", 
      "natural gas: ", round(E_ng, digits =3), " MWh (", round(E_ng/E_gen*100, digits =3), "%) \n", 
      "coal: ",  round(E_coal, digits =3), " MWh (", round(E_coal/E_gen*100, digits =3), "%) \n", 
      "nuclear: ",  round(E_nuclear, digits =3), " MWh (", round(E_nuclear/E_gen*100, digits =3), "%) \n", 
      "clean energy: ",  round(E_nuclear+E_solar+E_wind, digits =3), " MWh (", round((E_nuclear+E_solar+E_wind)/E_gen*100, digits =3), "%) \n", 
      "total generation: ", round(E_gen, digits =3), " MWh \n",
      "total load: ", round(E_load, digits =3), " MWh \n")

# labels = ["solar"; "wind"; "gas";"coal"; "nuclear"] 
# sizes = [E_solar; E_wind; E_ng; E_coal; E_nuclear]
# fig_energy = plot(labels, sizes,seriestype = :pie, title = "Energy Provider" )
# display(fig_energy)


# Planned Nodal Capacity of Generation Disrribution
print("Total solar installation cap: ", round(sum(w_PV)/1e3, digits =3)," (GW)\n")
print("Total wind installation cap: ", round(sum(w_WD)/1e3, digits =3)," (GW)\n")
print("Total energy storage installation cap: ", round(sum(w_ES)/1e3, digits =3)," (GWh)\n")

# figr = plot(w_WD/1e3, label= "wind")
# figr = plot!(w_PV/1e3, label= "solar", xlabel = "Node Index", ylabel = "Gen. Capacity (GW)",  title = "Installation Capacity Map")
# figr = plot!(w_ES/1e3, label= "Storage")
# display(figr)

# Output Solutions
print("The objective (total cost) is ", round(objective_value(model_plan),digits =3)," \n")
print("Investment Cost: ", round(value.(Obj_invest)/1e6,digits =3), " billion \$\n")
print("Solar Investment Cost: ", round(sum(c_PV_invest*(w_PV))/1e6,digits =3), " billion \$\n")
print("Wind Investment Cost: ", round(sum(c_WD_invest*w_WD)/1e6,digits =3), " billion \$\n")
print("Storage Investment Cost: ", round(sum(c_ES_invest*w_ES)/1e6,digits =3), " billion \$\n")
#print("Operational Cost: ", round(value.(Obj_opera),digits =3), " k\$\n")
