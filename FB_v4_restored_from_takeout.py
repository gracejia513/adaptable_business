# -*- coding: utf-8 -*-
"""
Created on Wed May  3 11:02:30 2023

@author: gracejia
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 21:13:46 2023
test a one dimensional restaurant where the business only operates takeout
@author: gracejia
"""

#%%
import pandas as pd
import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt
import random
import sys
import math
#%%
#set random seed
random.seed(10)
# output_path = "/Users/gracejia/Documents/A-UW/LEAP HI/Thrust2/local_git/leaphi_thrust2/output"
output_path = "D:\Grace\Thrust2"
seed = random.randint(1,100)
#%%
class FB():
    def __init__(self):
        '''
        This class is used to simulate the operation of a restaurant
        '''
        # self.tactics = {0:"Baseline",
        #                 1:"Tactic: Boost Productivity",
        #                 2:"Tactic: Adjust Quality",
        #                 3:"Tactics: Boost Productivity + Adjust Quality",
        #                 4:"Disable outdoor seating"}
        self.env_inf = {"employee_shortage": 60,
                        "supply_shock": 24,
                        "demand_shock": 48,
                        "production_decrease": 50,
                        "capacity_decrease" :100}
    # define different parameters to read in for simulating full-service restaurants and 
    # quick service restaurants

    def setParams(self,
                  channels = ["on_premise","off_premise"],
                  served   = ["on_premise","off_premise"],
                  working_hr = 8,
                  seed = 10,
                  fixed_employee = 21,
                  prod_inc = 0.05,
                  e_qa = -0.41,
                  A_discretize = 5
                  ):

        self.channels = channels
        self.served = served
        self.params = {}
        self.working_hr = working_hr
        self.seed = seed
        self.fixed_employee = fixed_employee
        self.prod_inc = prod_inc
        self.e_qa = e_qa
        self.A_discretize = A_discretize
        print("Status: General parameters set successfully")
        
    def getProduction_steady(self, e, s):
        # assuming a Cobb-Douglas production function
        # see an example here
        # https://mbounthavong.com/blog/2019/2/19/cobb-douglas-production-function-and-total-costs
        Q = self.A_picked * ((e * self.working_hr)**self.alpha) * (s**(1-self.alpha))
        return Q
    
    def setConstraintParams_new(self, 
                                file = False):
        # input cost is defined as the aggregated money needed to serve one customer
        # per the MyPlate requirement and the survey results, we assume that the average
        # input cost
        '''
        file: a dictionary that contains the parameter values
        QSR: a boolean that indicates whether the restaurant is a QSR or not
        '''

        if file:
            print("Parameter file passed successfully")
        else: # default is FSR
            file = {"demand": {"on_premise":171,
                               "off_premise": 65},
                    "default_supply":{"on_premise": 8.93,
                                      "off_premise": 8.93},
                    "labor_costs":{"on_premise":18,
                                   "off_premise":18},
                    "supply_costs":{"on_premise": 3.57,
                                    "off_premise": 3.57},
                    "fixed_costs": {"on_premise": 0,
                                    "off_premise": 616.67},
                    "price": {"on_premise": 20,
                              "off_premise": 35},
                    "capacity":  {"on_premise": 171,
                                  "off_premise": 65},
                    "full_quality":  {"on_premise": 8.93,
                                      "off_premise": 8.93},
                    "rent_utility": 333.33,
                    "A": 1.64,
                    "alpha": 0.7,
                    "employee_ub" :20*24, # originally this was 35*24 
                    "supply_ub" : 100, # originally this was 14832
                    "prod_ub" : 1.64,
                    
                    # disruptions
                    "employee_shortage" : 100, # 
                    "supply_shortage" : 100,
                    "demand_decrease" : {"on_premise": 100,
                                         "off_premise": 100},
                    "production_decrease" : {"on_premise": 100,
                                             "off_premise": 100,},
                    "capacity_decrease" : {"on_premise": 100,
                                           "off_premise": 100,},
                    "cost_e_increase" : 5.5,
                    "cost_s_increase"  : 8.8,
                    "delta" : {"on_premise": 1,
                               "off_premise": 0}}
                
                
        self.demand = file["demand"]
       # self.input_costs = file["input_costs"]
        self.labor_costs = file["labor_costs"]
        self.supply_costs = file["supply_costs"]
        self.fixed_costs = file["fixed_costs"]
        self.price = file["price"]
        self.capacity = file["capacity"]
        
        self.employee_ub = file["employee_ub"] 
        self.employee_ub_default = file["employee_ub"] 
        self.supply_ub = file["supply_ub"] 
        self.supply_ub_default = file["supply_ub"]
        self.prod_ub = file["prod_ub"]
        self.prod_ub_default = file["prod_ub"]
        self.capacity_default = file["capacity"]
        self.demand_default = file["demand"]
        self.default_supply = file["default_supply"]
        self.full_quality = file["full_quality"]
  
        self.O = file["rent_utility"]
        self.A = file["A"]
        self.alpha = file["alpha"]
        self.A_cost = 2000
        self.delta = file["delta"]
        
        self.employee_shortage = file["employee_shortage"] 
        self.supply_shortage = file["supply_shortage"]
        self.production_decrease = file["production_decrease"] 
        self.demand_decrease = file["demand_decrease"]
        self.capacity_decrease = file["capacity_decrease"]
        self.cost_e_increase = file["cost_e_increase"]
        self.cost_s_increase = file["cost_s_increase"]


        print("Status: Constraint parameters set successfully")


    def evalObj(self, result_dict):
        '''
        Evaluate the objective function    
        '''

        x = {"on_premise": result_dict['open[on_premise]'],
             "off_premise": result_dict['open[off_premise]']}
        e = {"on_premise": result_dict['work_hours[on_premise]'],
             "off_premise": result_dict['work_hours[off_premise]']}
        s = {"on_premise": result_dict['supply[on_premise]'],
             "off_premise": result_dict['supply[off_premise]']}   
        v = {"on_premise": result_dict['served[on_premise]'],
             "off_premise": result_dict['served[off_premise]']}
        a = result_dict['productivity_factor']                                           
        obj_val = -self.O + sum(- self.fixed_costs[i] * x[i] for i in self.channels) +\
                  -self.A_cost * (a - self.prod_ub) +\
                  sum(-self.labor_costs[i] * e[i] -self.supply_costs[i] * s[i]+ self.price[i] * v[i] for i in self.channels) 

        revenue = sum(self.price[i] * v[i] for i in self.channels)
        cost = revenue - obj_val
        profit_margin = obj_val / revenue  * 100
        food_purchase  = sum(self.supply_costs[i] * s[i] for i in self.channels)
        food_per = food_purchase / revenue * 100
        labor_cost = sum(self.labor_costs[i] * e[i] for i in self.channels)
        labor_per = labor_cost / revenue * 100
        operation_cost = sum(self.fixed_costs[i] * x[i] for i in self.channels)
        operation_cost_per = operation_cost / revenue * 100
        
        return {"objective_value":obj_val, 
                "profit_margin":profit_margin,
                "food_cost_percentage": food_per,
                "labor_cost_percentage": labor_per,
                "operation_cost_percentage":operation_cost_per}
      
    def getModel(self, initialize = 1, obj = "Pro_Max", tactic = 0):

        '''
        Build the optimization model 
        Decision variables
        X: whether an operation is open or not
        E: how many employee workhours are assigned to each operation
        S: how many supply are assigned to each operation
        V: how many customers are served by each operation   
        '''
        self.model = gp.Model(name = obj)
        
        # set parameters
        self.model.Params.Presolve = 2
        self.model.Params.seed = self.seed
        self.model.Params.NodeFileStart = 0.1
        self.model.Params.LogToConsole = 0
        self.model.params.NonConvex = 2
        
        # review the tactic dictionary
        # tactic_dict = {0:"Baseline",
        #               1:"Tactic: Boost Productivity",
        #               2:"Tactic: Adjust Quality",
        #               3:"Tactics: Adding Outdoor Capacity",
        #               4:"Tactics 1+2+3"}

        x = self.model.addVars([i for i in self.channels], vtype = gp.GRB.BINARY,name = "open")
        e = self.model.addVars([i for i in self.channels],lb = 0, ub = self.employee_ub, vtype = gp.GRB.CONTINUOUS, name = "work_hours")
        s = self.model.addVars([i for i in self.channels], lb = 0, ub = self.supply_ub, vtype = gp.GRB.CONTINUOUS, name = "supply")
        v = self.model.addVars([i for i in self.channels], lb = 0, vtype = gp.GRB.CONTINUOUS, name = "served")
           
        # additional decision variables needed when tactics are present in the model
        if tactic == 1 or tactic == 4: # need to boost efficiency
            a = self.model.addVar(vtype = gp.GRB.CONTINUOUS, lb = self.A,ub = self.A * 1.2, name = "productivity_factor")   
        else:
            a = self.model.addVar(vtype = gp.GRB.CONTINUOUS, lb = self.A,ub = self.A, name = "productivity_factor")
        # C: the average cost of one unit supply in operation i
        # P: the price of menu item in operation i 
        c = self.model.addVars([i for i in self.channels], lb = 0, vtype = gp.GRB.CONTINUOUS, name = "supply_cost")
        p = self.model.addVars([i for i in self.channels], lb = 0, vtype = gp.GRB.CONTINUOUS, name = "menu_price")
  
        # auxillary variables
        z_1 = self.model.addVar(vtype = gp.GRB.CONTINUOUS, name = "employee_aux_for_pf")
        z_2 = self.model.addVar(vtype = gp.GRB.CONTINUOUS, name = "supply_aux_for_pf" )
        q_12 = self.model.addVar(vtype = gp.GRB.CONTINUOUS, name = "aux_productivity_employee")
        a_help = self.model.addVar(vtype = gp.GRB.BINARY, name="prod_tactic_helper")
        print("Status: variables added.")   
        
        # set the objective function
        self.model.setObjective(-self.O -self.A_cost * a_help * (a - self.A) +\
                                sum(-self.fixed_costs[i] * x[i] -self.labor_costs[i] * e[i] -c[i] * s[i] +  p[i] * v[i] for i in self.channels), gp.GRB.MAXIMIZE)
 
        # self.model.setObjective(-self.O -self.A_cost * (a - self.A) +\
        #                         sum(-self.fixed_costs[i] * x[i] -self.labor_costs[i] * e[i] -c[i] * s[i] + p[i] * v[i] for i in self.channels ), 
        #                         gp.GRB.MAXIMIZE)

        print("Status: objective function set.")
     

        # Add constraints
        # if tactic == 1 or tactic == 4:
        self.model.addGenConstrIndicator(a_help, True, a - self.A >= 0.000001)
        # served <= demand in the base scenario = 0, 1, 4
        if tactic == 2 or tactic  == 4: # needs to adjust quality
            for i in self.channels:
                 change_in_q = (c[i] - self.supply_costs[i]) / self.supply_costs[i]
                 # served <= demand
                 self.model.addConstr(v[i] <= self.demand[i] * (1 + self.e_qa * change_in_q), "serve_less_than_adjusted_demand_" + i)
                 self.model.addConstr(p[i] == self.price[i] * (1 + change_in_q) , "adjust_menu_price_" + i)
          
        else:
            self.model.addConstrs((v[i] <= self.demand[i] for i in self.channels), "serve_less_than_demand" )
            for i in self.channels:
                self.model.addConstr(c[i] == self.supply_costs[i], "adjust_price_" + i)
                self.model.addConstr(p[i] == self.price[i], "adjust_menu_price_" + i)
        
        if tactic == 3 or tactic == 4: # need to add capacity for outdoor dining             
            self.delta['outdoor_dine_in'] = 1
            
        for i in self.channels:
            self.model.addConstr(x[i] <= self.delta[i], "whether_capacity_added_" + i) # available space   
            self.model.addConstr(v[i] <= self.capacity[i]*x[i], 'serve_less_than_capacity_' + i) # capacity constraint
        
        # helper functions/constraints for piecewise linear production function
        sum_e = self.model.addVar(vtype = gp.GRB.CONTINUOUS, lb = 0)
        sum_s = self.model.addVar(vtype = gp.GRB.CONTINUOUS, lb = 0)
        self.model.addConstr(sum_e == sum(e[i] for i in self.channels))
        self.model.addConstr(sum_s == sum(s[i] for i in self.channels))
        
        self.model.addConstr(sum_e <= self.employee_ub * sum(x[i] for i in self.channels), "hire_if_open" ) # labor constraint
        self.model.addConstr(sum_s <= self.supply_ub * sum(x[i] for i in self.channels), "buy_if_open") # buy if open
        self.model.addConstr(sum_e <= self.employee_ub, "hire_less_than_upperbound" ) # labor constraint
        self.model.addConstr(sum_s <= self.supply_ub, "buy_less_than_upperbound") # buy if open

    
        self.model.addGenConstrPow(sum_e, z_1, self.alpha, "employee_pow_helper_") # z_1 = sum(E[i]) ** self.alpha 
        self.model.addConstr(q_12 == a * z_1, "prod_emp_helper" ) # q_12 = A * z_1
        self.model.addGenConstrPow(sum_s, z_2, (1-self.alpha), "input_pow_helper") #z_2 = S **(1- alpha)
        # serve less than produced
        self.model.addConstr(sum(v[i] for i in self.channels) <= q_12 * z_2, "serve_less_than_produced_" )

        # open at least one mode
        self.model.addConstr(sum(x[i] for i in self.channels) >= 1, "open_at_least_one_mode")

         
        # self.model.update()
        print("Status: Constraints added.")
        
        self.model.params.NonConvex = 2
        
        # Optimize model
        self.model.optimize()
        
        # Print Objective function value
        if self.model.status == 3 :
            print("The problem is infeasible.")
            return [self.model.status, np.NaN, np.NaN]

        elif self.model.status == 4 :
            print("The problem is infeasible.")
            return [self.model.status, np.NaN, np.NaN]
        else:
            pass
      
        result_dict = {}
        self.quality_picked = {}
        for i in self.channels:
            self.quality_picked[i] = (c[i] - self.supply_costs[i]) / self.supply_costs[i]
            result_dict[i] = {"open":x[i].X,
                              "employee_work_hour":e[i].X,
                              "supply":s[i].X,
                              "served":v[i].X,
                              "efficiency_boost_factor":(a.X - a.LB)/a.LB,
                              "quality_factor":self.quality_picked[i].getValue()}
       
        self.result_dict_handmade = result_dict
        self.A_picked = a.X

        mvars = self.model.getVars()
        test_names = self.model.getAttr('VarName', mvars)
        test_values = self.model.getAttr('X', mvars)
        result = dict(zip(test_names, test_values))
        result["quality_factor"] = self.quality_picked
        self.result_dict = result
        
        return [self.model.status, self.model.ObjVal, self.result_dict]

    def summary(self, save = False):
        if self.model.status == 3:
            pass
        elif self.model.status == 4:
            pass
        else:
            print("=========Summary of the Optimization Problem ==========")
            print("Variables:")
            print(pd.DataFrame(self.result_dict_handmade))
            print("Objective function value = {}".format(self.model.ObjVal))
            # employee_hours = self.model.getVarByName("work_hours").X 
            # print("Employee hours = {}".format(self.model.getVarByName("work_hours")))
            # print("Supply purchased = {}".format(self.model.getVarByName("supply")))

            curr_prod = []
            # employee = self.model.getVars()[4].X
            employee = self.model.getVars()[1].X
            for i in range(0,2):
                employee = self.model.getVars()[i+2].X
                supply = self.model.getVars()[i+4].X
                curr_prod.append(self.getProduction_steady(employee ,supply)) 
                print("Employee work hour assigned for {} = {}".format(self.channels[i],employee) )
                print("Supply purchased for {} = {}".format(self.channels[i],supply) )
                print("\n")
                print("Current production = {}".format(curr_prod[i]))

    def runEnv(self, scenario = "employee_shortage", mode ="on_premise", step = 2, min_val = 0, manual = False, tactic = 0):
        '''
        scenario: "employee_shortage", "supply_shock", "demand_shock", "production_decrease", "capacity_decrease"
        mode: "indoor_dine_in", "outdoor_dine_in", "take_out", "delivery"
        step: step size of the loop
        min_val: disruption severity where the simulation starts
        manual: if manual is given, the simulation will run until the manual value
        tactic_dict = {0:"Baseline (no tactics)",
                       1:"Tactic = Boost Productivity",
                       2:"Tactic = Adjust Quality",
                       3:"Tactic = Adding Outdoor Capacity",
                       4:"Tactics 1+2+3"}
        '''
        result_dict = {}
        result_dict["status"] = {}
        result_dict["values"] = {}
        result_dict["variables"] = {}
        result_dict["model"] = {}
        if scenario == "employee_shortage":
            if manual:
                max_disruption = manual
            else:
                max_disruption = self.employee_shortage
                
            for i in range(min_val, max_disruption + step, step):
                self.employee_ub = self.employee_ub_default * (-1) * (i / 100 - 1)

                for j in self.channels:
                    #self.labor_costs[j] += self.labor_costs[j] * self.cost_e_increase / 100 # originally the labor cost increases linearly as employees are getting harder to hire
                     self.labor_costs[j] += self.labor_costs[j] * (self.cost_e_increase / 100) * math.pow((i - min_val) / step, 2) #this line increases the labor cost quadratically. 
                model_results = self.getModel(tactic = tactic)
                self.model.update()
                self.summary()
                result_dict["status"][i] = model_results[0]
                result_dict["values"][i] = model_results[1]
                result_dict["variables"][i] = model_results[2]
                result_dict["model"][i] = self.model
            return result_dict
             
        if scenario == "supply_shock":
            if manual:
                max_disruption = manual
            else:
                max_disruption = self.supply_shortage
            for i in range(min_val, max_disruption + step, step):
                self.supply_ub = self.supply_ub * (-1) * (i / 100 - 1)
                for j in self.channels: 
                   # self.supply_costs[j] += self.supply_costs[j] * self.cost_s_increase / 100 # originally the supply cost increases linearly
                    self.supply_costs[j] += self.supply_costs[j] * (self.cost_s_increase / 100) * math.pow((i - min_val) / step, 2) #this line increases the supply cost quadratically. 
                model_results = self.getModel(tactic = tactic)
                self.summary()
                result_dict["status"][i] = model_results[0]
                result_dict["values"][i] = model_results[1]
                result_dict["variables"][i] = model_results[2]
                result_dict["model"][i] = self.model
            return result_dict
        
        if scenario == "demand_shock":
            if manual:
                max_disruption = manual
            else:
                max_disruption = self.demand_decrease
            y = mode
            result_dict["status"][y] = {}
            result_dict["values"][y] = {}
            result_dict["variables"][y] = {}
            for i in range(min_val, max_disruption[y] + step, step):
                for x in self.channels:
                    self.demand[x] = self.demand_default[x] * (-1) * (i / 100 - 1)
                model_results = self.getModel(tactic = tactic)
                self.summary()
                result_dict["status"][y][i] = model_results[0]
                result_dict["values"][y][i] = model_results[1]
                result_dict["variables"][y][i] = model_results[2]
                result_dict["model"][i] = self.model
            return result_dict
         
        if scenario == "production_decrease":
            if manual:
                max_disruption = manual
            else:
                max_disruption = self.production_decrease
            y = mode
            result_dict["status"][y] = {}
            result_dict["values"][y] = {}
            result_dict["variables"][y] = {}
            for i in range(min_val, max_disruption[y], step):               
                self.A = self.prod_ub_default * (-1) * (i / 100 - 1)
                model_results = self.getModel(tactic = tactic)
                self.summary()
                result_dict["status"][y][i] = model_results[0]
                result_dict["values"][y][i] = model_results[1]
                result_dict["variables"][y][i] = model_results[2]
                result_dict["model"][i] = self.model
            return result_dict 

                   
        if scenario == "capacity_decrease":
            if manual:
                max_disruption = manual
            else:
                max_disruption = self.capacity_decrease
            y = mode
            result_dict["status"][y] = {}
            result_dict["values"][y] = {}
            result_dict["variables"][y] = {}
            for i in range(min_val, max_disruption[y] + step, step):
                for x in self.channels:
                    self.capacity[x] = self.capacity[x] * (-1) * (i / 100 - 1)
                model_results = self.getModel(tactic = tactic)
                self.summary()
                result_dict["status"][y][i] = model_results[0]
                result_dict["values"][y][i] = model_results[1]
                result_dict["variables"][y][i] = model_results[2]
                result_dict["model"][i] = self.model
            return result_dict
#%% FB_vis_new

class FB_vis_new():
    def __init__(self, env_inf = {"Employee Shock": 60,
                                  "Supply Shock": 23,
                                  "Demand Shock": 48,
                                  "Capacity Shock": 100,
                                  "Production Shock" :26},
                 n_bkpts = 3,
                 mode_bank = ["indoor_dine_in", "outdoor_dine_in", "takeout", "delivery"],
                 output_path = "D:\Grace\Thrust2"
                 ):
        self.case_bank = ['obj_vs_disruption','decision_vs_disruption']
        self.mode_bank = mode_bank
        self.env_inf = env_inf
        self.n_bkpts = n_bkpts -1
        self.output_path = output_path
        self.tactics = {0: "baseline case",
                        1: "boost productivity",
                        2: "adjust quality",
                        3: "adding outdoor capacity",
                        4: "tactic-enabled case"}
        
    def vis(self, result_dict, title, case = 'obj_vs_disruption', mode = False, step = 2):
        if mode:
            x = np.array(list(result_dict["values"][mode].keys()))
            y = np.array(list(result_dict["values"][mode].values()))

        else:
            x = np.array(list(result_dict["values"].keys()))
            y = np.array(list(result_dict["values"].values()))

        col = np.where(y is np.nan, 'k', np.where(y < 0, 'r','b'))
        # identify system infeasible region
        infeasible_set = np.where(np.isnan(y))
        if len(infeasible_set[0]) == 0:
            system_infeasible = 105
        else:
            system_infeasible = np.where(np.isnan(y))[0][0] * step
            
        # identify profit infeasibile region
        profit_infeasible = 101
        loc = 0
        for i in range(0,len(y)):
            
            if y[i] < 0:
                profit_infeasible = loc * step
                break
            else:
                loc += 1
        
        # identify envirnment infeasibility region
        environment_infeasible = self.env_inf[title]
        
        if case == 'obj_vs_disruption':
            if mode:
                x = np.array(list(result_dict["values"][mode].keys()))
                y = np.array(list(result_dict["values"][mode].values()))

            else:
                x = np.array(list(result_dict["values"].keys()))
                y = np.array(list(result_dict["values"].values()))

            col = np.where(y is np.nan, 'k', np.where(y < 0, 'r','b'))
            # identify system infeasible region
            infeasible_set = np.where(np.isnan(y))
            if len(infeasible_set[0]) == 0:
                system_infeasible = 105
            else:
                system_infeasible = np.where(np.isnan(y))[0][0] * step
                
            # identify profit infeasibile region
            profit_infeasible = 101
            loc = 0
            for i in range(0,len(y)):

                if y[i] < 0:
                    profit_infeasible = loc * step
                    break
                else:
                    loc += 1
            
            # identify envirnment infeasibility region
            environment_infeasible = self.env_inf[title]
            
            plt.figure()
            plt.scatter(x,y, c = 'gray',s = [2]*len(x), label = 'Profit')
            if system_infeasible <= 100:
                plt.axvline(x=system_infeasible, color='k', linewidth=1.0, linestyle='--', label = "System Infeasibility") 
            if profit_infeasible <= 100:
                plt.axvline(x = profit_infeasible, color = 'g', linewidth = 1.0, linestyle = ':', label = "Profit Infeasibility")
            plt.axvline(x = environment_infeasible, color = 'purple', linewidth = 1.0, linestyle = 'dotted', label = "Environment Infeasibility")           
            plt.legend()
            plt.title(title)
            plt.xlabel("Severity of disruption (%)")
            plt.ylabel("Business Profit")
            plt.xlim(right = 100)
           # plt.show()
            
            analysis = {}
            analysis["system_infeasibility"] = system_infeasible
            analysis["profit_infeasibility"] = profit_infeasible
            analysis["environment_infeasibility"] = environment_infeasible
            
            return analysis
        
        if case == 'decision_vs_disruption':
            if mode:
                pass
            else:
                pass
            
            # find out where the restaurant closes its operation
            closed = {}
            help_status = []
            for i in self.mode_bank:
                closed[i] = 105
            for i in range(0,len(result_dict)):
                for k in self.mode_bank:
                    help_status[k] = []
                    for j in range(0,len(result_dict[i]['variables'])):
                        help_status[k].append(result_dict[i]['variables'][j]["open["+k+ "]"])

            y_emp = []
            y_mode = {}
            y_mode["indoor_dine_in"] =[]
            y_mode["outdoor_dine_in"] =[]
            y_mode["takeout"]=[]
            y_mode["delivery"]=[]
            for i in x[:system_infeasible]:
                # num of employee_hr
                y_val = result_dict["model"][i].getVarByName("sum_of_employee_hours").X 
                y_emp.append(y_val)
                for key in ["indoor_dine_in", "outdoor_dine_in", "takeout", "delivery"]:
                    open_or_not = result_dict["model"][i].getVarByName("open["+key+']').X 
                    y_mode[key].append(open_or_not)
            #plot
            fig,ax = plt.subplots()
            ax.plot(x[:len(y_emp)],y_emp, c = 'gray', label = 'Employee Hours')
            ax.set_ylabel("Employee Hours")
            # ax2 = ax.twinx()

            locs = {}
            locs["indoor_dine_in"] =[]
            locs["outdoor_dine_in"] =[]
            locs["takeout"]=[]
            locs["delivery"]=[]
            for key in self.mode_bank:               
                # ax2.plot(x[:len(y_emp)],y_mode[key], label = "Availability of " + key)
                closed = np.where(np.array(y_mode[key]) ==0)[0]
                if len(closed):
                    locs[key].append(np.where(np.array(y_mode[key]) ==0)[0][0])
                else:
                    locs[key].append(-10)
            for i in range(0,len(self.mode_bank)):
                key = self.mode_bank[i]
                ax.annotate('   '+ key +' closed', xy=(locs[key][0], 0+0.05 ),
                            xytext = (locs[key][0], 50+i*20),
                            arrowprops=dict(facecolor='black', shrink=0.05), horizontalalignment='left', verticalalignment='top')
            if system_infeasible <= 100:
                plt.axvline(x=system_infeasible, color='k', linewidth=1.0, linestyle='--', label = "System Infeasibility") 
            if profit_infeasible <= 100:
                plt.axvline(x = profit_infeasible, color = 'g', linewidth = 1.0, linestyle = ':', label = "Profit Infeasibility")
            plt.axvline(x = environment_infeasible, color = 'purple', linewidth = 1.0, linestyle = 'dotted', label = "Environment Infeasibility")
                
            
            plt.legend()
            plt.title(title)
            plt.xlabel("Severity of disruption (%)")
            # plt.ylabel("Decision Variables")
            plt.xlim(right = system_infeasible + 5)
            #plt.show()    
             
            analysis = {}
            analysis["mode_availability"] = locs
            analysis["employee_hr"] = y_emp
            analysis["system_infeasibility"] = system_infeasible
            analysis["profit_infeasibility"] = profit_infeasible
            analysis["environment_infeasibility"] = environment_infeasible    

            return analysis
       
        
    def check_slack(self,result_dict,mode, pts_x,pts_y):
        slack = {}
        if mode:
            pass
        else:
            for i in pts_x:
                model = result_dict["model"][i]
                num_constrs = model.NumConstrs
                for j in range(0,num_constrs):
                    slack[i][model.getConstrs()[j].ConstrName] = model.slack[j]
                
    def check_IIS(self,IIS_model):
        # First we look for all the IIS constraints
        print("Computing IIS...")
        IIS_model.computeIIS()
        if IIS_model.IISMinimal:
            print("IIS is minimal.\n")
        else:
            print("IIS is NOT minimal. \n")
        
        print("The following constraint(s) cannot be satisfied:\n")
        print("Linear Constraints:")
        for c in IIS_model.getConstrs():
            if c.IISConstr:
                print("%s"%c.ConstrName)
        print("Q Constraints:")
        for c in IIS_model.getQConstrs():
            if c.IISQconstr:
                print("%s"%c.ConstrName)
        print("General Constraints:")
        for c in IIS_model.getGenConstrs():
            if c.IISGenConstr:
                print("%s"%c.ConstrName)
        
        # Then we loop until reduce to a model that can be solved
        # make a copy
        m = IIS_model
        removed = []
        while True:
            m.computeIIS()
            for c in m.getConstrs():
                if c.IISConstr:
                    print("%s"%c.ConstrName)
                    removed.append(str(c.ConstrName))
                    m.remove(c)
                    break
                
            m.optimize()
            status = m.status
            
            if status == gp.GRB.UNBOUNDED:
                print("Model cannot be solved because it is unbounded")
                sys.exit(0)
            if status == gp.GRB.OPTIMAL:
                break
            if status != gp.GRB.INF_OR_UNBD and status != gp.GRB.INFEASIBLE:
                print("optimization stopped with status %d" % status)
                sys.exit(0)
            
            # print('The following constraints were removed to get a feasible LP:\n')
        return m,removed
            
    def relax(self, m,constr_pen = False):
        # this m should be a copy of an infeasible model
        print("try relaxing the constraints.")
        orignumvars = m.NumVars
        origvars = m.getVars()
        constrs = m.getConstrs()
        # with constraint penalty:
        if constr_pen:
            m.feasRelax(1,True, origvars, None, None, constrs, constr_pen)
            m.optimize()
            
        else:
            m.feasRelaxS(1,False,False, True)
            m.optimize()
            status = m.status
            if status in (gp.GRB.INF_OR_UNBD, gp.GRB.INFEASIBLE, gp.GRB.UNBOUNDED):
                print("the relaxed model cannot be solved because it is infeasible or unbounded.")
                sys.exit(1)
            if status != gp.GRB.OPTIMAL:
                print("optimization was stopped with status %d" %status)
                sys.exit(1)
        
        print("Slack values:")
        slacks = m.getVars()[orignumvars:]
        for sv in slacks:
            if sv.X > 1e-6:
                print('%s = %g' % (sv.VarName, sv.X))
        
        return m

    def vis_mult(self, title, result_dicts, mode = False, step = 1,tactics = [0], theme = "Employee_shortage", case = "obj_vs_disruption"):
        if len(result_dicts) == 1:
            analysis = self.vis(result_dict = result_dicts[0], title = title, case = case, mode = mode )
            return analysis
        else:
            tactic_dict = {0:"Baseline case",
                          1:"Tactic: Boost Productivity",
                          2:"Tactic: Adjust Quality",
                          3:"Tactic: Adding Outdoor Capacity",
                          4:"Tactic-enabled case"}
            infeasibilities = {}
            x_list = []
            y_list = []
            count = 0
            c = ["gray","salmon","gold","lightblue","teal"]
            for result_dict in result_dicts:
                if mode:
                    x_list.append(np.array(list(result_dict["values"][mode].keys())))
                    y_list.append(np.array(list(result_dict["values"][mode].values())))
                else:
                    x_list.append(np.array(list(result_dict["values"].keys())))
                    y_list.append(np.array(list(result_dict["values"].values())))
                
                if count == 0:
                    # identify system infeasible region
                    infeasible_set = np.where(np.isnan(y_list[count]))
                    if len(infeasible_set[0]) == 0:
                        system_infeasible = 105
                    else:
                        system_infeasible = np.where(np.isnan(y_list[count]))[0][0] * step
                        
                    # identify profit infeasibile region
                    profit_infeasible = 101
                    loc = 0
                    for i in range(0,len(y_list[count])):                        
                        if y_list[count][i] < 0:
                            profit_infeasible = loc * step
                            break
                        else:
                            loc += 1
                    # identify envirnment infeasibility region
                    environment_infeasible = self.env_inf[title]
                else:
                    infeasible_set = np.where(np.isnan(y_list[count]))
                    if len(infeasible_set[0]) == 0:
                        infeasibilities[count] = 105
                    else:
                        infeasibilities[count] = np.where(np.isnan(y_list[count]))[0][0] * step
                count += 1    
        
        plt.figure(figsize = (15,8))
        ax1 = plt.subplot(2, 1, 1)
        for i in range(0,len(x_list)):
            plt.scatter(x_list[i],y_list[i], c = c[i], label = tactic_dict[tactics[i]])
        if system_infeasible <= 100:
            plt.axvline(x=system_infeasible, color='k', linewidth=2.0, linestyle='--', label = "System Infeasibility (baseline)") 
        if profit_infeasible <= 100:
            plt.axvline(x = profit_infeasible, color = 'g', linewidth = 2.0, linestyle = ':', label = "Profit Infeasibility (baseline)")
        plt.axvline(x = environment_infeasible, color = 'purple', linewidth = 2.0, linestyle = 'dotted', label = "Environment Infeasibility")           
        plt.legend(loc = "upper right")
        plt.title(title, fontsize=20)
        plt.xlabel("Severity of disruption (%)", fontsize=12)
        plt.ylabel("Business Profit", fontsize=12)
        plt.xlim(right = 100)
        plt.savefig(output_path + "/Profit trajectory: " + title)
        plt.show()
        
        # # calculate return /
        # ax2 = plt.subplot(2, 1, 2)
        # returns = []
        # for i in range(0, len(x_list)-1):
        #     growth = []
        #     for j in range(0,len(y_list[i+1][0:system_infeasible])):
        #         new = y_list[i+1][j]
        #         old = y_list[0][j]
        #         growth.append(new - old )
        #     returns.append(growth)
        #     plt.scatter(x_list[i+1][0:system_infeasible],growth, c = c[i+1], label =  tactic_dict[i+1])           
        # plt.legend()
        # plt.xlabel("Disruption Severity (%)", fontsize=12)
        # plt.ylabel("Profit growth compared to no tactic", fontsize=12)
        # plt.xlim(right = 100)
        # plt.savefig(output_path + "/Profit Trajectory and Return of Tactics " + theme + ".png")
        # plt.show()     

#%%
fb = FB()
fb.setParams()
fb.setConstraintParams_new()
#%%
fb.getModel(tactic = 4)
#%% run scenarios
tactics = [0,1,2,3,4]
#%%
employee_result_dicts = []
for tactic in tactics:
    fb = FB()
    fb.setParams()
    fb.setConstraintParams_new()
    employee_result_dicts.append(fb.runEnv(scenario = "employee_shortage", step = 1, tactic = tactic))

for tactic in tactics:
    df = pd.DataFrame.from_dict(employee_result_dicts[tactic])
    df.to_csv(output_path + "/Employee_result_tactic_" + str(tactic)+".csv")
#%%
fb_vis = FB_vis_new()
emp_analysis = fb_vis.vis_mult(title = "Employee Shock", result_dicts = employee_result_dicts, tactics = tactics)
#%%
supply_result_dicts = []
for tactic in tactics:
    fb = FB()
    fb.setParams()
    fb.setConstraintParams_new()
    supply_result_dicts.append(fb.runEnv(scenario = "supply_shock", step = 1, tactic = tactic))
# export employee_result_dicts
#%%
for tactic in tactics:
    df = pd.DataFrame.from_dict(supply_result_dicts[tactic])
    df.to_csv(output_path + "/Supply_result_tactic_" + str(tactic)+".csv")
#%%
fb_vis = FB_vis_new()
supply_analysis = fb_vis.vis_mult(title = "Supply Shock", theme = "Supply Shock",result_dicts = supply_result_dicts, tactics = tactics)
#%%
#  #%% scenario: capacity_decrease
capacity_result_dicts = []
for tactic in tactics:
    fb = FB()
    fb.setParams()
    fb.setConstraintParams_new()
    capacity_result_dicts.append(fb.runEnv(scenario = "capacity_decrease", step = 1,mode = "on_premise", tactic = tactic))
# export capacity_result_dicts
for tactic in tactics:
    df = pd.DataFrame.from_dict(capacity_result_dicts[tactic])
    df.to_csv(output_path + "/Capacity_result_tactic_" + str(tactic)+".csv")
# fb_vis = FB_vis_new()
# capacity_analysis = fb_vis.vis_mult(title = "Capacity Decrease", theme = "Capacity Decrease",result_dicts = capacity_result_dicts,mode = "on_premise", tactics = tactics)

# #%% scenario: demand shock
demand_result_dicts = []
for tactic in tactics:
    fb = FB()
    fb.setParams()
    fb.setConstraintParams_new()
    demand_result_dicts.append(fb.runEnv(scenario = "demand_shock", step = 1,mode = "on_premise", tactic = tactic))
# export employee_result_dicts
for tactic in tactics:
    df = pd.DataFrame.from_dict(demand_result_dicts[tactic])
    df.to_csv("Demand_result_tactic_" + str(tactic)+".csv")

# fb_vis = FB_vis_new()
# demand_analysis = fb_vis.vis_mult(title = "Demand Decrease",theme = "Demand Decrease", result_dicts = demand_result_dicts,mode = "on_premise", tactics = tactics)
# #%%    
# #%% scenario:production_decrease
prod_result_dicts = []
for tactic in tactics:
    fb = FB()
    fb.setParams()
    fb.setConstraintParams_new()
    prod_result_dicts.append(fb.runEnv(scenario = "production_decrease", step = 1,mode = "on_premise", tactic = tactic))
# export employee_result_dicts
for tactic in tactics:
    df = pd.DataFrame.from_dict(prod_result_dicts[tactic])
    df.to_csv(output_path + "/Prod_result_tactic_" + str(tactic)+".csv")

#%%
employee_condensed = []
employee_condensed.append(employee_result_dicts[0])
employee_condensed.append(employee_result_dicts[4])
fb_vis = FB_vis_new()
emp_analysis = fb_vis.vis_mult(title = "Employee Shock", result_dicts = employee_condensed, tactics = [0,4])
#%%
supply_condensed = []
supply_condensed.append(supply_result_dicts[0])
supply_condensed.append(supply_result_dicts[4])
fb_vis = FB_vis_new()
supply_analysis = fb_vis.vis_mult(title = "Supply Shock", result_dicts = supply_condensed, tactics = [0,4])
#%%
capacity_condensed = []
capacity_condensed.append(capacity_result_dicts[0])
capacity_condensed.append(capacity_result_dicts[4])
fb_vis = FB_vis_new()
capacity_analysis = fb_vis.vis_mult(title = "Capacity Shock", result_dicts = capacity_condensed, 
                                    mode = "on_premise", tactics = [0,4])
#%%
prod_condensed = []
prod_condensed.append(prod_result_dicts[0])
prod_condensed.append(prod_result_dicts[4])
fb_vis = FB_vis_new()
prod_analysis = fb_vis.vis_mult(title = "Production Shock", result_dicts = prod_condensed, 
                                mode = "on_premise", tactics = [0,4])
#%%
demand_condensed = []
demand_condensed.append(demand_result_dicts[0])
demand_condensed.append(demand_result_dicts[4])
fb_vis = FB_vis_new()
demand_analysis = fb_vis.vis_mult(title = "Demand Shock", result_dicts = demand_condensed, 
                                mode = "on_premise", tactics = [0,4])

#%% looking closer at the decision variables (no tactic)
# [employee_result_dicts,supply_result_dicts,capacity_result_dicts,
#  prod_result_dicts,demand_result_dicts]

df_interest = supply_result_dicts
work_hour_1 = []
work_hour_4 = []
supply_count_1 = []
supply_count_4 = []
efficiency_1 = []
efficiency_4 = []
quality_1 = []
quality_4 = []
price_1 = []
price_4 = []
open_status_offpremise_1 = [] 
open_status_offpremise_4 = [] 
served_1 = []
served_4 = []
severity = []
for i in range(0,100,2):
    severity.append(i)
    # open_status_takeout.append(employee_result_dicts[0]['model'][i].getVars()[0].X)
    curr_served_1 = df_interest[0]['model'][i].getVars()[6].X + df_interest[0]['model'][i].getVars()[7].X
    curr_served_4 = df_interest[4]['model'][i].getVars()[6].X + df_interest[4]['model'][i].getVars()[7].X
    served_1.append(curr_served_1)
    served_4.append(curr_served_4)
    
    if curr_served_1 <1:
        work_hour_1.append(0)
        supply_count_1.append(0)
        efficiency_1.append(0)
        quality_1.append(0)
        price_1.append(0)
    else:
        work_hour_1.append(df_interest[0]['model'][i].getVars()[2].X + df_interest[0]['model'][i].getVars()[3].X)
        supply_count_1.append(df_interest[0]['model'][i].getVars()[4].X + df_interest[0]['model'][i].getVars()[5].X)
        efficiency_1.append(df_interest[0]['model'][i].getVars()[8].X)
        quality_1.append(df_interest[0]['model'][i].getVars()[9].X)
        price_1.append(df_interest[0]['model'][i].getVars()[11].X)
    
    open_status_offpremise_1.append(df_interest[0]['model'][i].getVars()[1].X)
        
    if curr_served_4 < 1:
        work_hour_4.append(0)
        supply_count_4.append(0)
        efficiency_4.append(0)
        quality_4.append(0)
        price_4.append(0)
    else:
        work_hour_4.append(df_interest[4]['model'][i].getVars()[2].X + df_interest[4]['model'][i].getVars()[3].X)
        supply_count_4.append(df_interest[4]['model'][i].getVars()[4].X + df_interest[4]['model'][i].getVars()[5].X)
        efficiency_4.append(df_interest[4]['model'][i].getVars()[8].X)
        quality_4.append(df_interest[4]['model'][i].getVars()[9].X)
        price_4.append(df_interest[4]['model'][i].getVars()[11].X)

    open_status_offpremise_4.append(df_interest[4]['model'][i].getVars()[1].X)
    
#%%
x_label = 'supply shock severity'
# create a 2x2 grid of subplots
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

# plot the first subplot
axs[0, 0].plot(severity, supply_count_1, color='gray', label='baseline')
axs[0, 0].plot(severity, supply_count_4, color='salmon', label='tactic-enabled')
axs[0, 0].set_xlabel(x_label, fontsize=14)
axs[0, 0].set_ylabel('supply purchased', fontsize=14)
axs[0, 0].legend(loc='upper right', fontsize='large')

# plot the second subplot
colors = ['#CFA79D', '#97A5C0']
axs[0, 1].plot(severity, efficiency_4, color=colors[0], label='efficiency')
axs[0, 1].set_xlabel(x_label, fontsize = 14)
axs[0, 1].set_ylabel('efficiency', fontsize = 14)
axs[0, 1].ticklabel_format(axis='y', style='plain')
axs2 = axs[0, 1].twinx()
axs2.plot(severity, quality_4, color=colors[1], label='quality')
axs2.set_ylabel('quality', fontsize = 14)
axs[0, 1].legend(frameon=False,loc = 'lower left',fontsize='large')
axs2.legend(frameon=False,loc = 'upper left',fontsize='large')
# ax1.legend(frameon=False, bbox_to_anchor=(0, 0), loc='lower left')
# ax2.legend(frameon=False, bbox_to_anchor=(0, 0.1), loc='lower left')
# axs[0, 1].legend()

# plot the third subplot
axs[1, 0].plot(severity, served_1, color=colors[0], label='baseline')
axs[1, 0].plot(severity, served_4, color=colors[1], label='tactic-enabled')
axs[1, 0].set_xlabel(x_label, fontsize=14)
axs[1, 0].set_ylabel('customer served', fontsize=14)
axs[1, 0].legend(loc='best', fontsize='large')

# plot the fourth subplot
axs[1, 1].plot(severity, price_1, color=colors[0], label='baseline')
axs[1, 1].plot(severity, price_4, color=colors[1], label='tactic-enabled')
axs[1, 1].set_xlabel(x_label, fontsize=14)
axs[1, 1].set_ylabel('menu price', fontsize=14)
axs[1, 1].legend(loc='best', fontsize='large')

plt.tight_layout()
plt.show()

#%% work_hour
plt.figure(figsize = (10,6))
# create the first line plot on the first set of y-axes
plt.plot(severity, work_hour_1, color='gray', label = 'baseline')
plt.plot(severity, work_hour_4, color='salmon', label = 'tactic-enabled')
plt.xlabel('employee shock severity', fontsize = 14)
plt.ylabel('work hours', fontsize = 14)
plt.legend(loc = 'upper right', fontsize='large')
plt.show()

#%% looking at the tactics
colors = ['#CFA79D', '#97A5C0']
fig, ax1 = plt.subplots()

# create the first line plot on the first set of y-axes
ax1.plot(severity, efficiency_4, color=colors[0], label = 'efficiency')
ax1.set_xlabel('employee shock severity')
ax1.set_ylabel('efficiency', color=colors[0])
# ax1.set_ylim(1.63, 1.97)
ax1.ticklabel_format(axis='y', style='plain')

# create the second set of y-axes that shares the same x-axis with the first set of y-axes
ax2 = ax1.twinx()

# create the second line plot on the second set of y-axes
ax2.plot(severity, quality_4, color=colors[1], label = 'quality')
ax2.set_ylabel('quality', color=colors[1])

ax1.legend(loc='best')
ax2.legend(loc='best')
# optionally, add a legend for each line plot
# ax1.legend(frameon=False, bbox_to_anchor=(0, 0), loc='lower left')
# ax2.legend(frameon=False, bbox_to_anchor=(0, 0.1), loc='lower left')
# plt.title('Checking tactic adjust quality')
plt.show()
#%% looking closer at the decision variables (served customer)
colors = ['#CFA79D', '#97A5C0']

plt.figure(figsize = (8,4))
plt.plot(severity, served_1, color=colors[0], label = 'baseline')
plt.plot(severity, served_4, color=colors[1], label = 'tactic-enabled')
plt.xlabel('employee shock severity', fontsize = 14)
plt.ylabel('customer served', fontsize = 14)
plt.legend(loc = 'upper right', fontsize='large')

# plt.title('Checking tactic adjust quality')
plt.show()

#%%
colors = ['#CFA79D', '#97A5C0']

plt.figure(figsize = (8,4))
# create the first line plot on the first set of y-axes
plt.plot(severity, price_1, color=colors[0], label = 'baseline')
plt.plot(severity, price_4, color=colors[1], label = 'tactic-enabled')
plt.xlabel('employee shock severity', fontsize = 14)
plt.ylabel('menu price', fontsize = 14)
plt.legend(loc = 'best', fontsize='large')

# plt.title('Checking tactic adjust quality')
plt.show()

#%%
#%% looking closer at the decision variables (no tactic)
df_interest = prod_result_dicts
work_hour_1 = []
work_hour_4 = []
supply_count_1 = []
supply_count_4 = []
efficiency_1 = []
efficiency_4 = []
quality_1 = []
quality_4 = []
price_1 = []
price_4 = []
# open_status_takeout = [] checked that takeout always opens
# open_status_outdoor = [] checked that outdoor never opens.
served_1 = []
served_4 = []
severity = []
for i in range(0,100,2):
    severity.append(i)
    # open_status_takeout.append(employee_result_dicts[0]['model'][i].getVars()[0].X)
    work_hour_1.append(df_interest[0]['model'][i].getVars()[2].X + df_interest[0]['model'][i].getVars()[3].X)
    work_hour_4.append(df_interest[4]['model'][i].getVars()[2].X + df_interest[4]['model'][i].getVars()[3].X)
    supply_count_1.append(df_interest[0]['model'][i].getVars()[4].X + df_interest[0]['model'][i].getVars()[5].X)
    supply_count_4.append(df_interest[4]['model'][i].getVars()[4].X + df_interest[4]['model'][i].getVars()[5].X)
    efficiency_1.append(df_interest[0]['model'][i].getVars()[8].X)
    efficiency_4.append(df_interest[4]['model'][i].getVars()[8].X)
    quality_1.append(df_interest[0]['model'][i].getVars()[9].X)
    quality_4.append(df_interest[4]['model'][i].getVars()[9].X)
    served_1.append(df_interest[0]['model'][i].getVars()[6].X + df_interest[0]['model'][i].getVars()[7].X)
    served_4.append(df_interest[4]['model'][i].getVars()[6].X + df_interest[4]['model'][i].getVars()[7].X)
    price_1.append(df_interest[0]['model'][i].getVars()[11].X)
    price_4.append(df_interest[4]['model'][i].getVars()[11].X)


#%%
plt.figure(figsize = (10,6))
# create the first line plot on the first set of y-axes
plt.plot(severity, work_hour_1, color='gray', label = 'quality')
plt.plot(severity, work_hour_4, color='salmon', label = 'tactic-enabled')
plt.xlabel('employee shock severity', fontsize = 14)
plt.ylabel('work hours', fontsize = 14)
plt.legend(loc = 'upper right', fontsize='large')
plt.show()

#%% looking at the tactics
colors = ['#CFA79D', '#97A5C0']
fig, ax1 = plt.subplots()

# create the first line plot on the first set of y-axes
ax1.plot(severity, efficiency_4, color=colors[0], label = 'efficiency')
ax1.set_xlabel('employee shock severity')
ax1.set_ylabel('efficiency', color=colors[0])
ax1.set_ylim(0, 1.97)
ax1.ticklabel_format(axis='y', style='plain')

# create the second set of y-axes that shares the same x-axis with the first set of y-axes
ax2 = ax1.twinx()

# create the second line plot on the second set of y-axes
ax2.plot(severity, quality_4, color=colors[1], label = 'quality')
ax2.set_ylabel('quality', color=colors[1])

# optionally, add a legend for each line plot
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
# optionally, add a legend for each line plot
# ax1.legend(frameon=False, bbox_to_anchor=(0, 0), loc='lower left')
# ax2.legend(frameon=False, bbox_to_anchor=(0, 0.1), loc='lower left')
# plt.title('Checking tactic adjust quality')
plt.show()
#%% looking closer at the decision variables (work hour and served customer)
colors = ['#CFA79D', '#97A5C0']

plt.figure(figsize = (8,4))
# create the first line plot on the first set of y-axes
plt.plot(severity, served_1, color=colors[0], label = 'baseline')
plt.plot(severity, served_4, color=colors[1], label = 'tactic-enabled')
plt.xlabel('employee shock severity', fontsize = 14)
plt.ylabel('customer served', fontsize = 14)
plt.legend(loc = 'upper right', fontsize='large')

# plt.title('Checking tactic adjust quality')
plt.show()

#%%
colors = ['#CFA79D', '#97A5C0']

plt.figure(figsize = (8,4))
# create the first line plot on the first set of y-axes
plt.plot(severity, price_1, color=colors[0], label = 'baseline')
plt.plot(severity, price_4, color=colors[1], label = 'tactic-enabled')
plt.xlabel('employee shock severity', fontsize = 14)
plt.ylabel('menu price', fontsize = 14)
plt.legend(loc = 'upper right', fontsize='large')

# plt.title('Checking tactic adjust quality')
plt.show()