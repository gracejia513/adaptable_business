#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 20:49:37 2022

@author: gracejia
"""
#%%
import pandas as pd
import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt
import random
import sys
#%%
#set random seed
random.seed(10)
# output_path = "/Users/gracejia/Documents/A-UW/LEAP HI/Thrust2/local_git/leaphi_thrust2/output"
output_path = "D:\Grace\Thrust2"
#%%
seed = random.randint(1,100)
#%%
class FB():
    def __init__(self):
        '''
        This class is used to simulate the operation of a restaurant
        '''
        # self.sce_bank = ["normal",
        #                  "employee_shortage",
        #                  "supply_shock",
        #                  "demand_shock",
        #                  "production_decrease",
        #                  "capacity_shock",
        #                  ]
        # self.sce = "steady_state"
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
                  variable_size = [4,4,4,4],
                  channels = ["indoor_dine_in","outdoor_dine_in","takeout","delivery"],
                  input_e  = ["employee"],
                  input_s  = ["supply"],
                  served   = ["indoor_dine_in","outdoor_dine_in","takeout","delivery"],
                  price_adj = 3,
                  fixed_adj = 5,
                  working_hr = 8,
                  seed = 10,
                  fixed_employee = 21,
                  prod_inc = 0.05,
                  e_qa = -0.41,
                  ):
        '''
        variable_size: a list of integers that indicate the number of variables in each channel
        channels: a list of strings that indicate the operation mode the restaurant has
        input_e: a list of strings that indicate the input variables that are employees
        input_s: a list of strings that indicate the input variables that are supplies
        served: a list of strings that indicate the number of customers served
        price_adj: a float that indicates the price adjustment
        fixed_adj: a float that indicates the fixed cost adjustment
        working_hr: a float that indicates the working hours per day
        seed: an integer that indicates the random seed
        fixed_employee: an integer that indicates the number of fixed employees
        prod_inc: a float that indicates the increase in productivity
        e_qa: a float that indicates the elasticity of quality w.r.t. employees
        '''
        self.channels = channels
        self.input_e = input_e
        self.input_s = input_s
        self.served = served
        self.variable_size = variable_size
        self.params = {}
        self.price_adj = price_adj
        self.fixed_adj = fixed_adj 
        self.working_hr = working_hr
        self.seed = seed
        self.fixed_employee = fixed_employee
        self.prod_inc = prod_inc
        self.e_qa = e_qa
        print("Status: General parameters set successfully")
        
    def getProduction_steady(self, e, s):
        # assuming a Cobb-Douglas production function
        # see an example here
        # https://mbounthavong.com/blog/2019/2/19/cobb-douglas-production-function-and-total-costs
        Q = self.A_picked * ((e * self.working_hr)**self.alpha) * (s**(1-self.alpha))
        return Q
    
    def setConstraintParams_new(self, 
                                file = False,
                                QSR = False,):
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
            file = {"demand": {"indoor_dine_in": 171,
                               "outdoor_dine_in": 20,
                               "takeout":14,
                               "delivery": 31},
                    "default_supply":{"indoor_dine_in": 8.93,
                                       "outdoor_dine_in": 8.93,
                                       "takeout": 8.93,
                                       "delivery": 8.93},
                    "labor_costs":{"indoor_dine_in": 18,
                                    "outdoor_dine_in": 18,
                                    "takeout":18,
                                    "delivery": 18},
                    "supply_costs":{"indoor_dine_in": 3.57,
                                    "outdoor_dine_in": 3.57,
                                    "takeout": 3.57,
                                    "delivery": 3.57},
                    "fixed_costs": {"indoor_dine_in": 0,
                                       "outdoor_dine_in": 616.67,
                                       "takeout": 0,
                                       "delivery": 0},
                    "price": {"indoor_dine_in": 20,
                                  "outdoor_dine_in": 20,
                                  "takeout": 30,
                                  "delivery": 40,},
                    "capacity":  {"indoor_dine_in": 171,
                                     "outdoor_dine_in": 20,
                                     "takeout": 14,
                                     "delivery": 31,},
                    "full_quality":  {"indoor_dine_in": 8.93,
                                     "outdoor_dine_in": 8.93,
                                     "takeout": 8.93,
                                     "delivery": 8.93,},
                    "rent_utility": 333.33,
                    "A": 1.64,
                    "alpha": 0.7,
                    "employee_ub" :35*24,
                    "supply_ub" : 14832,
                    "prod_ub" : 1.64,
                    
                    # disruptions
                    "employee_shortage" : 100, # 
                    "supply_shortage" : 100,
                    "demand_decrease" : {"indoor_dine_in": 100,
                                         "outdoor_dine_in": 100,
                                         "takeout": 100,
                                         "delivery": 100,},
                    "production_decrease" : {"indoor_dine_in": 100,
                                             "outdoor_dine_in": 100,
                                             "takeout": 100,
                                             "delivery": 100,},
                    "capacity_decrease" : {"indoor_dine_in": 100,
                                           "outdoor_dine_in": 100,
                                           "takeout": 100,
                                           "delivery": 100,},
                    "cost_e_increase" : 5.5,
                    "cost_s_increase"  : 8.8,
                    "delta" : {"indoor_dine_in": 1,
                               "outdoor_dine_in": 0,
                               "takeout": 1,
                               "delivery": 1}}

        if QSR:
            file = {"demand": {"indoor_dine_in": 31,
                               "outdoor_dine_in": 30*0.5,
                               "takeout": 30*1.5,
                               "delivery": 16*1.5},
                    "input_costs": {"indoor_dine_in": {"employee": 17* self.working_hr /4,
                                                      "supply": 31 *1.2 * (1/3) * ( 2 * 0.7 + 5.5* 0.3 + 2.5*1 + 3*0.22 + 6*0.09) * self.price_adj},
                                    "outdoor_dine_in":{"employee": 17* self.working_hr / 4,
                                                      "supply": 0.5*31 *1.2 * (1/3) * ( 2 * 0.7 + 5.5* 0.3 + 2.5*1 + 3*0.22 + 6*0.09) * self.price_adj},
                                    "takeout": {"employee": 15*self.working_hr / 4,
                                               "supply": 1.5*31 * 1.2 * (1/3) * ( 2 * 0.7 + 5.5* 0.3 + 2.5*1 + 3*0.22 + 6*0.09)* self.price_adj },
                                    "delivery": {"employee": 17*self.working_hr / 4,
                                                "supply": 1.5*16 * 1.2* (1/3) * ( 2 * 0.7 + 5.5* 0.3 + 2.5*1 + 3*0.22 + 6*0.09) * self.price_adj},},
                    "fixed_costs": {"indoor_dine_in": 70 * self.fixed_adj,
                                       "outdoor_dine_in": 70 * self.fixed_adj,
                                       "takeout": 70 * self.fixed_adj,
                                       "delivery": 70 * self.fixed_adj,},
                    "price": {"indoor_dine_in": 20,
                                  "outdoor_dine_in": 20,
                                  "takeout": 20,
                                  "delivery": 30,},
                    "capacity":  {"indoor_dine_in": 31,
                                     "outdoor_dine_in": 30*0.5,
                                     "takeout": 30*1.5,
                                     "delivery": 16*1.5,},
                    "A": 0.7,
                    "alpha": 0.7,
                    "employee_ub" : 20,
                    "supply_ub" : 250,
                    "prod_ub" : 0.7,
                    
                    # disruptions
                    "employee_shortage" : 100,
                    "supply_shortage" : 100,
                    "demand_decrease" : {"indoor_dine_in": 100,
                                         "outdoor_dine_in": 100,
                                         "takeout": 100,
                                         "delivery": 100,},
                    "production_decrease" : {"indoor_dine_in": 100,
                                             "outdoor_dine_in": 100,
                                             "takeout": 100,
                                             "delivery": 100,},
                    "capacity_decrease" : {"indoor_dine_in": 100,
                                           "outdoor_dine_in": 100,
                                           "takeout": 100,
                                           "delivery": 100,},
                    "cost_e_increase" : 5.5,
                    "cost_s_increase"  : 8.8,
                    }
                
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
        
        self.A = file["A"]
        self.alpha = file["alpha"]
        
        self.employee_shortage = file["employee_shortage"] 
        self.supply_shortage = file["supply_shortage"]
        self.production_decrease = file["production_decrease"] 
        self.demand_decrease = file["demand_decrease"]
        self.capacity_decrease = file["capacity_decrease"]
        self.cost_e_increase = file["cost_e_increase"]
        self.cost_s_increase = file["cost_s_increase"]
    
                                
        self.demand = file["demand"]
        # self.input_costs = file["input_costs"]
        self.fixed_costs = file["fixed_costs"]
        self.price = file["price"]
        self.capacity = file["capacity"]
        self.default_supply = file["default_supply"]
        self.full_quality = file["full_quality"]
        
        self.employee_ub = file["employee_ub"] 
        self.employee_ub_default = file["employee_ub"] 
        self.supply_ub = file["supply_ub"] 
        self.supply_ub_default = file["supply_ub"]
        self.prod_ub = file["prod_ub"]
        self.prod_ub_default = file["prod_ub"]
        self.capacity_default = file["capacity"]
        self.demand_default = file["demand"]
        self.operational_cost = file["rent_utility"]
        self.A = file["A"]
        self.alpha = file["alpha"]
        self.A_cost = 2000
        
        self.employee_shortage = file["employee_shortage"] 
        self.supply_shortage = file["supply_shortage"]
        self.production_decrease = file["production_decrease"] 
        self.demand_decrease = file["demand_decrease"]
        self.capacity_decrease = file["capacity_decrease"]
        self.cost_e_increase = file["cost_e_increase"]
        self.cost_s_increase = file["cost_s_increase"]
        self.delta = file["delta"]
        print("Status: Constraint parameters set successfully")


    def evalObj(self, result_dict):
        '''
        Evaluate the objective function    
        '''

        x = {"indoor_dine_in" : result_dict['open[indoor_dine_in]'],
             "outdoor_dine_in" : result_dict['open[outdoor_dine_in]'],
             "takeout" : result_dict['open[takeout]'],
             "delivery" : result_dict['open[delivery]']}
                                      
        e = {"indoor_dine_in" : result_dict['work_hours[indoor_dine_in]'],
             "outdoor_dine_in" : result_dict['work_hours[outdoor_dine_in]'],
             "takeout" : result_dict['work_hours[takeout]'],
             "delivery" : result_dict['work_hours[delivery]']}  
        
        s = {"indoor_dine_in" : result_dict['supply[indoor_dine_in]'],
             "outdoor_dine_in" : result_dict['supply[outdoor_dine_in]'],
             "takeout" : result_dict['supply[takeout]'],
             "delivery" : result_dict['supply[delivery]']}   
                                      
        v = {"indoor_dine_in" : result_dict['served[indoor_dine_in]'],
             "outdoor_dine_in" : result_dict['served[outdoor_dine_in]'],
             "takeout" : result_dict['served[takeout]'],
             "delivery" : result_dict['served[delivery]']}  
        a = result_dict['productivity_factor']                                           
        obj_val = -self.operational_cost + sum(- self.fixed_costs[i] * x[i] for i in self.channels) +\
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
        
        # calculate production dictionary
        #self.getProduction_new()
        
        # review the tactic dictionary
        # tactic_dict = {0:"Baseline",
        #               1:"Tactic: Boost Productivity",
        #               2:"Tactic: Adjust Quality",
        #               3:"Tactics: Adding Outdoor Capacity",
        #               4:"Tactics 1+2+3"}

        X = self.model.addVars([i for i in self.channels], vtype = gp.GRB.BINARY,name = "open")
        E = self.model.addVars([i for i in self.channels],lb = 0, ub = self.employee_ub, vtype = gp.GRB.CONTINUOUS, name = "work_hours")
        S = self.model.addVars([i for i in self.channels], lb = 0, ub = self.supply_ub, vtype = gp.GRB.CONTINUOUS, name = "supply")
        V = self.model.addVars([i for i in self.channels], lb = 0, vtype = gp.GRB.CONTINUOUS, name = "served")
           
        # additional decision variables needed when tactics are present in the model
        if tactic == 1 or tactic == 4: # need to boost efficiency
            A = self.model.addVar(vtype = gp.GRB.CONTINUOUS, lb = self.A,ub = self.A * 1.2, name = "productivity_factor")
        else:
            A = self.model.addVar(vtype = gp.GRB.CONTINUOUS, lb = self.A,ub = self.A, name = "productivity_factor")
        # C: the average cost of one unit supply in operation i
        # P: the price of menu item in operation i 
        C = self.model.addVars([i for i in self.channels ], lb = 0, vtype = gp.GRB.CONTINUOUS, name = "supply_cost")
        P = self.model.addVars([i for i in self.channels], lb = 0, vtype = gp.GRB.CONTINUOUS, name = "menu_price")
  
        # auxillary variables
        # w_1 = self.model.addVar(vtype = gp.GRB.INTEGER, name = "sum_of_employee_hours")
        z_1 = self.model.addVars([i for i in self.channels], vtype = gp.GRB.CONTINUOUS, name = "employee_aux_for_pf")
        z_2 = self.model.addVars([i for i in self.channels], vtype = gp.GRB.CONTINUOUS, name = "supply_aux_for_pf" )
        q_12 = self.model.addVars([i for i in self.channels], vtype = gp.GRB.CONTINUOUS, name = "aux_productivity_employee")
        print("Status: variables added.")        
        # set the objective function
        self.model.setObjective(-self.operational_cost -self.A_cost * (A - self.prod_ub) +\
                                sum(-self.fixed_costs[i] * X[i] -self.labor_costs[i] * E[i] -C[i] * S[i] + P[i] * V[i] for i in self.channels ), 
                                gp.GRB.MAXIMIZE)

        print("Status: objective function set.")
     

        # Add constraints
        # served <= demand in the base scenario = 0, 1, 4
        if tactic == 2 or tactic  == 4: # needs to adjust quality
            for i in self.channels:
                change_in_q = (C[i] - self.supply_costs[i] ) / self.supply_costs[i]
                # suppose the relationship between change in quality and change in demand is linear
                # D_i = m * change_in_q + b passes through (0,D_i) and (-1,0), then
                # m = D_i / (1 - 0) = D_i, b = D_i
                self.model.addConstr(V[i] <= self.demand[i] * change_in_q + self.demand[i], "serve_less_than_adjusted_demand_" + i)              
                self.model.addConstr(P[i] == self.price[i] * (1 + change_in_q), "adjust_menu_price_" + i)
        else:
            self.model.addConstrs((V[i] <= self.demand[i] for i in self.channels), "serve_less_than_demand" )
            for i in self.channels:
                self.model.addConstr(C[i] == self.supply_costs[i], "adjust_price_" + i)
                self.model.addConstr(P[i] == self.price[i], "adjust_menu_price_" + i)
        
        if tactic == 3 or tactic == 4: # need to add capacity for outdoor dining             
            self.delta['outdoor_dine_in'] = 1
            
        for i in self.channels:
            self.model.addConstr(X[i] <= self.delta[i], "whether_capacity_added_" + i) # available space   
            self.model.addConstr(V[i] <= self.capacity[i]*X[i], 'serve_less_than_capacity_' + i) # capacity constraint
            self.model.addConstr(E[i] <= self.employee_ub * X[i], "hire_less_than_upperbound_" + i) # labor constraint
            self.model.addConstr(S[i] >= X[i], "buy_if_open_" + i) # buy if open
            # helper functions/constraints for piecewise linear production function
            self.model.addGenConstrPow(E[i], z_1[i], self.alpha, "employee_pow_helper_" + i) # z_1 = E[i] ** self.alpha 
            self.model.addConstr(q_12[i] == A * z_1[i], "prod_emp_helper" ) # q_12[i] = A * z_1[i]
            self.model.addGenConstrPow(S[i], z_2[i], (1-self.alpha), "input_pow_helper" + i) #z_2[i] = S[i] **(1- alpha)
            # serve less than produced
            self.model.addConstr(V[i] <= q_12[i] * z_2[i], "serve_less_than_produced_" + i)

        # open at least one mode
        self.model.addConstr(sum(X[i] for i in self.channels) >= 1, "open_at_least_one_mode")
        self.model.addConstr(sum(S[i] for i in self.channels) <= self.supply_ub * sum(X[i] for i in self.channels), "don't buy too much")
        self.model.addConstr(sum(S[i] for i in self.channels) <= self.supply_ub, "buy_less_than_upperbound")
         
        self.model.update()
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
            self.quality_picked[i] = (C[i] - self.supply_costs[i]) / self.supply_costs[i]
            result_dict[i] = {"open":X[i].X,
                              "supply":S[i].X,
                              "served":V[i].X,
                              "efficiency_boost_factor":(A.X - A.LB)/A.LB,
                              "quality_factor":self.quality_picked[i].getValue()}
       
        self.result_dict_handmade = result_dict
        self.A_picked = A.X

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
            for i in range(0,4):
                employee = self.model.getVars()[i+4].X
                supply = self.model.getVars()[i+8].X
                curr_prod.append(self.getProduction_steady(employee ,supply)) 
                print("Employee work hour assigned for {} = {}".format(self.channels[i],employee) )
                print("Supply purchased for {} = {}".format(self.channels[i],supply) )
            print("\n")
            print("Current production = {}".format(curr_prod))

    def runEnv(self, scenario = "employee_shortage", mode ="indoor_dine_in", step = 2, min_val = 0, manual = False, tactic = 0):
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
                    self.labor_costs[j] += self.labor_costs[j] * self.cost_e_increase / 100 
                model_results = self.getModel(tactic = tactic)
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
                    self.supply_costs[j] += self.supply_costs[j] * self.cost_s_increase / 100                
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
                self.A = self.prod_ub * (-1) * (i / 100 - 1)
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
#%% 5 visualization setup

# import ruptures as rpt
# import jenkspy

class FB_vis():
    def __init__(self, env_inf = {"Employee Shortage": 60,
                                  "Supply Shock": 24,
                                  "Demand Shock": 48,
                                  "Capacity Shock": 100,
                                  "Production Shock" :50},
                 n_bkpts = 3,
                 mode_bank = ["indoor_dine_in", "outdoor_dine_in", "takeout", "delivery"]
                 ):
        self.case_bank = ['obj_vs_disruption','decision_vs_disruption']
        self.mode_bank = mode_bank
        self.env_inf = env_inf
        self.n_bkpts = n_bkpts -1
        
    def vis(self, result_dict, title, case = 'obj_vs_disruption', mode = False, step = 2):
        if mode:
            x = np.array(list(result_dict["values"][mode].keys()))
            y = np.array(list(result_dict["values"][mode].values()))
            # y = np.array(list(result_dict["values"][mode].values()) )
            # for i in range(len(y)):
            #     y[i] = -y[i]
        else:
            x = np.array(list(result_dict["values"].keys()))
            y = np.array(list(result_dict["values"].values()))
            # y = np.array(list(result_dict["values"].values()))
            # for i in range(len(y)):
            #     y[i] = -y[i]
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
            # analysis["chg_pts_x"] = chg_pts_x
            # analysis["chg_pts_y"] = chg_pts_y
            analysis["system_infeasibility"] = system_infeasible
            analysis["profit_infeasibility"] = profit_infeasible
            analysis["environment_infeasibility"] = environment_infeasible
            
            return analysis
        
        if case == 'decision_vs_disruption':
            # get num of employee
            # if mode:
            #     x = np.array(list(result_dict["values"][mode].keys()))
            #     y = np.array(list(result_dict["values"][mode].values()))
            # else:
            #     x = np.array(list(result_dict["model"].keys()))
            #     # y = np.array(list(result_dict["model"][i].getVars()[14].X for i in range(0,len(x))))
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
            tactic_dict = {0:"Baseline",
                          1:"Tactic: Boost Productivity",
                          2:"Tactic: Adjust Quality",
                          3:"Tactic: Adding Outdoor Capacity",
                          4:"Tactics 1+2+3"}
            infeasibilities = {}
            x_list = []
            y_list = []
            count = 0
            c = ["gray","salmon","gold","lightblue","teal"]
            for result_dict in result_dicts:
                if mode:
                    x_list.append(np.array(list(result_dicts[result_dict]["values"][mode].keys())))
                    y_list.append(np.array(list(result_dicts[result_dict]["values"][mode])))
                else:
                    x_list.append(np.array(list(result_dicts[result_dict]["values"].keys())))
                    y_list.append(np.array(list(result_dicts[result_dict]["values"])))
                
                if count == 0:
                    # col = np.where(y_list[count] is np.nan, 'k', np.where(y_list[count] < 0, 'r','b'))
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
            plt.scatter(x_list[i],y_list[i], c = c[i], label = tactic_dict[i])
        if system_infeasible <= 100:
            plt.axvline(x=system_infeasible, color='k', linewidth=2.0, linestyle='--', label = "System Infeasibility (no tactics)") 
        if profit_infeasible <= 100:
            # axs[0].axvline(x = profit_infeasible, color = 'g', linewidth = 1.0, linestyle = ':', label = "Profit Infeasibility (no tactics)")
            plt.axvline(x = profit_infeasible, color = 'g', linewidth = 2.0, linestyle = ':', label = "Profit Infeasibility (no tactics)")
        plt.axvline(x = environment_infeasible, color = 'purple', linewidth = 2.0, linestyle = 'dotted', label = "Environment Infeasibility (no tactics)")           
        plt.legend()
        plt.title(title, fontsize=20)
        # plt.xlabel("Severity of disruption (%)", fontsize=12)
        plt.ylabel("Business Profit", fontsize=12)
        plt.xlim(right = 100)
        # plt.savefig("Profit trajectory: " + theme)
        #plt.show()
        
        # calculate return /
        # plt.figure(figsize = (15,8))
        ax2 = plt.subplot(2, 1, 2)
        returns = []
        for i in range(0, len(x_list)-1):
            growth = []
            for j in range(0,len(y_list[i+1][0:system_infeasible])):
                new = y_list[i+1][j]
                old = y_list[0][j]
                growth.append(abs(new - old) )
            # growth = [100 * abs((y - x)) / abs(x) for x, y in zip(y_list[i+1][0:system_infeasible], y_list[0][0:system_infeasible])]

            returns.append(growth)
            plt.scatter(x_list[i+1][0:system_infeasible],growth, c = c[i+1], label =  tactic_dict[i+1])
            # plt.axvline(x=infeasibilities[i+1], color= c[i+1], linewidth=1.0, linestyle='--', label = "System Infeasibility for " + tactic_dict[i+1] )
        
        # if system_infeasible <= 100:
        #     plt.axvline(x=system_infeasible, color='k', linewidth=1.0, linestyle='--', label = "System Infeasibility (no tactics)") 
        # if profit_infeasible <= 100:
        #     plt.axvline(x = profit_infeasible, color = 'g', linewidth = 1.0, linestyle = ':', label = "Profit Infeasibility (no tactics)")
        # plt.axvline(x = environment_infeasible, color = 'purple', linewidth = 1.0, linestyle = 'dotted', label = "Environment Infeasibility (no tactics)")           
        plt.legend()
        # plt.title(title, fontsize=20)
        plt.xlabel("Disruption Severity (%)", fontsize=12)
        plt.ylabel("Profit growth compared to no tactic", fontsize=12)
        plt.xlim(right = 100)
        plt.savefig("Profit Trajectory and Return of Tactics: " + theme)
        # return returns
       # plt.show()             /
                    # return analysis
      
         

#%% test FB()
# fb = FB()
# fb.setParams(seed = seed)
# fb.setConstraintParams_new(QSR = False)
# #%%
# fb.getModel(tactic = 0)
# fb.summary()
# # #%%
# # # test objective value: sanity check
# sanity_check = fb.evalObj(fb.result_dict)  
            
#%% FB_vis_new
class FB_vis_new():
    def __init__(self, env_inf = {"Employee Shortage": 60,
                                  "Supply Shock": 24,
                                  "Demand Decrease": 48,
                                  "Capacity Decrease": 100,
                                  "Production Decrease" :50},
                 n_bkpts = 3,
                 mode_bank = ["indoor_dine_in", "outdoor_dine_in", "takeout", "delivery"],
                 output_path = "D:\Grace\Thrust2"
                 ):
        self.case_bank = ['obj_vs_disruption','decision_vs_disruption']
        self.mode_bank = mode_bank
        self.env_inf = env_inf
        self.n_bkpts = n_bkpts -1
        self.output_path = output_path
        
    def vis(self, result_dict, title, case = 'obj_vs_disruption', mode = False, step = 2):
        if mode:
            x = np.array(list(result_dict["values"][mode].keys()))
            y = np.array(list(result_dict["values"][mode].values()))
            # y = np.array(list(result_dict["values"][mode].values()) )
            # for i in range(len(y)):
            #     y[i] = -y[i]
        else:
            x = np.array(list(result_dict["values"].keys()))
            y = np.array(list(result_dict["values"].values()))
            # y = np.array(list(result_dict["values"].values()))
            # for i in range(len(y)):
            #     y[i] = -y[i]
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
            # analysis["chg_pts_x"] = chg_pts_x
            # analysis["chg_pts_y"] = chg_pts_y
            analysis["system_infeasibility"] = system_infeasible
            analysis["profit_infeasibility"] = profit_infeasible
            analysis["environment_infeasibility"] = environment_infeasible
            
            return analysis
        
        if case == 'decision_vs_disruption':
            # get num of employee
            # if mode:
            #     x = np.array(list(result_dict["values"][mode].keys()))
            #     y = np.array(list(result_dict["values"][mode].values()))
            # else:
            #     x = np.array(list(result_dict["model"].keys()))
            #     # y = np.array(list(result_dict["model"][i].getVars()[14].X for i in range(0,len(x))))
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
            tactic_dict = {0:"Baseline",
                          1:"Tactic: Boost Productivity",
                          2:"Tactic: Adjust Quality",
                          3:"Tactic: Adding Outdoor Capacity",
                          4:"Tactics 1+2+3"}
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
                    # col = np.where(y_list[count] is np.nan, 'k', np.where(y_list[count] < 0, 'r','b'))
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
            plt.scatter(x_list[i],y_list[i], c = c[i], label = tactic_dict[i])
        if system_infeasible <= 100:
            plt.axvline(x=system_infeasible, color='k', linewidth=2.0, linestyle='--', label = "System Infeasibility (no tactics)") 
        if profit_infeasible <= 100:
            # axs[0].axvline(x = profit_infeasible, color = 'g', linewidth = 1.0, linestyle = ':', label = "Profit Infeasibility (no tactics)")
            plt.axvline(x = profit_infeasible, color = 'g', linewidth = 2.0, linestyle = ':', label = "Profit Infeasibility (no tactics)")
        plt.axvline(x = environment_infeasible, color = 'purple', linewidth = 2.0, linestyle = 'dotted', label = "Environment Infeasibility")           
        plt.legend()
        plt.title(title, fontsize=20)
        # plt.xlabel("Severity of disruption (%)", fontsize=12)
        plt.ylabel("Business Profit", fontsize=12)
        plt.xlim(right = 100)
        # plt.savefig("Profit trajectory: " + theme)
        # plt.show()
        
        # calculate return /
        # plt.figure(figsize = (15,8))
        ax2 = plt.subplot(2, 1, 2)
        returns = []
        for i in range(0, len(x_list)-1):
            growth = []
            for j in range(0,len(y_list[i+1][0:system_infeasible])):
                new = y_list[i+1][j]
                old = y_list[0][j]
                growth.append(new - old )
            # growth = [100 * abs((y - x)) / abs(x) for x, y in zip(y_list[i+1][0:system_infeasible], y_list[0][0:system_infeasible])]

            returns.append(growth)
            plt.scatter(x_list[i+1][0:system_infeasible],growth, c = c[i+1], label =  tactic_dict[i+1])
            # plt.axvline(x=infeasibilities[i+1], color= c[i+1], linewidth=1.0, linestyle='--', label = "System Infeasibility for " + tactic_dict[i+1] )
        
        # if system_infeasible <= 100:
        #     plt.axvline(x=system_infeasible, color='k', linewidth=1.0, linestyle='--', label = "System Infeasibility (no tactics)") 
        # if profit_infeasible <= 100:
        #     plt.axvline(x = profit_infeasible, color = 'g', linewidth = 1.0, linestyle = ':', label = "Profit Infeasibility (no tactics)")
        # plt.axvline(x = environment_infeasible, color = 'purple', linewidth = 1.0, linestyle = 'dotted', label = "Environment Infeasibility (no tactics)")           
        plt.legend()
        # plt.title(title, fontsize=20)
        plt.xlabel("Disruption Severity (%)", fontsize=12)
        plt.ylabel("Profit growth compared to no tactic", fontsize=12)
        plt.xlim(right = 100)
        plt.savefig(output_path + "/Profit Trajectory and Return of Tactics " + theme + ".png")
        # return returns
        plt.show()             
                    # return analysis
#%% test run

fb = FB()
fb.setParams()  
fb.setConstraintParams_new()
# supply_result = fb.runEnv(scenario ="supply_shock",step = 1)
emp_result = fb.runEnv(scenario = "employee_shortage")
#%% test_visualization
tactics = [0]
fb_vis = FB_vis_new()
# sup_analysis = fb_vis.vis(title = "Supply Shock", result_dict = supply_result) 
emp_analysis = fb_vis.vis(title = "Employee Shortage", result_dict = emp_result)   
#%% compare with QSR, still has error, proceed to QSR.py for analysis
# # scenario: employee_shortage
# tactics = [0,4]
# employee_result_dicts = []
# for tactic in tactics:
#     fb = FB()
#     fb.setParams()
#     fb.setConstraintParams_new(QSR = True)
#     employee_result_dicts.append(fb.runEnv(scenario = "employee_shortage", step = 1, tactic = tactic))
# # export employee_result_dicts
# for tactic in tactics:
#     df = pd.DataFrame.from_dict(employee_result_dicts[tactic])
#     df.to_csv(output_path + "/QSR_Employee_result_tactic_" + str(tactic)+".csv")
# #%%        
# fb_vis = FB_vis_new()
# emp_analysis = fb_vis.vis_mult(title = "Employee Shortage", result_dicts = employee_result_dicts, tactics = tactics)                 
#%%
# populate the multi-tactic result, tactics = 0,1,2,3,4
# scenario: employee_shortage
tactics = [3]
employee_result_dicts = []
for tactic in tactics:
    fb = FB()
    fb.setParams()
    fb.setConstraintParams_new()
    employee_result_dicts.append(fb.runEnv(scenario = "employee_shortage", step = 1, tactic = tactic))
#%%# export employee_result_dicts
for tactic in tactics:
    df = pd.DataFrame.from_dict(employee_result_dicts[tactic-3])
    df.to_csv(output_path + "/Employee_result_tactic_" + str(tactic)+".csv")

fb_vis = FB_vis_new()
emp_analysis = fb_vis.vis_mult(title = "Employee Shortage", result_dicts = employee_result_dicts, tactics = tactics)
#%%
# tactics = [0,1,2,3,4]
tactics = [0,1,3]
supply_result_dicts = []
for tactic in tactics:
    fb = FB()
    fb.setParams()
    fb.setConstraintParams_new()
    supply_result_dicts.append(fb.runEnv(scenario = "supply_shock", step = 1, tactic = tactic))
# export employee_result_dicts
for tactic in tactics:
    if tactic == 0 or tactic == 1:
        df = pd.DataFrame.from_dict(supply_result_dicts[tactic])
    else:
        df = pd.DataFrame.from_dict(supply_result_dicts[tactic-1])
    df.to_csv(output_path + "/Supply_result_tactic_" + str(tactic)+".csv")

fb_vis = FB_vis_new()
supply_analysis = fb_vis.vis_mult(title = "Supply Shock", theme = "Supply Shock",result_dicts = supply_result_dicts, tactics = tactics)
#%%
#  #%% scenario: capacity_decrease
# tactics = [0,1,2,3,4]
tactics = [0,1,3]
capacity_result_dicts = []
for tactic in tactics:
    fb = FB()
    fb.setParams()
    fb.setConstraintParams_new()
    capacity_result_dicts.append(fb.runEnv(scenario = "capacity_decrease", step = 1,mode = "indoor_dine_in", tactic = tactic))
# export employee_result_dicts
for tactic in tactics:
    if tactic == 0 or tactic == 1:
        df = pd.DataFrame.from_dict(supply_result_dicts[tactic])
    else:
        df = pd.DataFrame.from_dict(supply_result_dicts[tactic-1])
    df.to_csv(output_path + "/Capacity_result_tactic_" + str(tactic)+".csv")
fb_vis = FB_vis_new()
capacity_analysis = fb_vis.vis_mult(title = "Capacity Decrease", theme = "Capacity Decrease",result_dicts = capacity_result_dicts,mode = "indoor_dine_in", tactics = tactics)
# #%
# #%% scenario: demand shock
# tactics = [0,1,2,3,4]
tactics = [0,1,3]
demand_result_dicts = []
for tactic in tactics:
    fb = FB()
    fb.setParams()
    fb.setConstraintParams_new()
    demand_result_dicts.append(fb.runEnv(scenario = "demand_shock", step = 1,mode = "indoor_dine_in", tactic = tactic))
# export employee_result_dicts
for tactic in tactics:
    if tactic == 0 or tactic == 1:
        df = pd.DataFrame.from_dict(supply_result_dicts[tactic])
    else:
        df = pd.DataFrame.from_dict(supply_result_dicts[tactic-1])
    df.to_csv("Demand_result_tactic_" + str(tactic)+".csv")

fb_vis = FB_vis_new()
Demand_analysis = fb_vis.vis_mult(title = "Demand Decrease",theme = "Demand Decrease", result_dicts = demand_result_dicts,mode = "indoor_dine_in", tactics = tactics)
# #%%    
# #%% scenario:production_decrease
# tactics = [0,1,2,3,4]
tactics = [0,1,3]
prod_result_dicts = []
for tactic in tactics:
    fb = FB()
    fb.setParams()
    fb.setConstraintParams_new()
    prod_result_dicts.append(fb.runEnv(scenario = "production_decrease", step = 1,mode = "indoor_dine_in", tactic = tactic))
# export employee_result_dicts
for tactic in tactics:
    if tactic == 0 or tactic == 1:
        df = pd.DataFrame.from_dict(supply_result_dicts[tactic])
    else:
        df = pd.DataFrame.from_dict(supply_result_dicts[tactic-1])
    df.to_csv(output_path + "/Prod_result_tactic_" + str(tactic)+".csv")


fb_vis = FB_vis_new()
prod_analysis = fb_vis.vis_mult(title = "Production Decrease", theme = "Production Decrease",result_dicts = prod_result_dicts,mode = "indoor_dine_in", tactics = tactics)
#%%

#%%

# fb = FB()
# fb.setParams()  
# fb.setConstraintParams_new()
# supply_result = fb.runEnv(scenario ="supply_shock",step = 1)
# #%%
# fb = FB()
# fb.setParams()  
# fb.setConstraintParams_new()
# demand_result_indoor = fb.runEnv(scenario ="demand_shock", mode = 'indoor_dine_in', step = 1)
# #%%
# fb = FB()
# fb.setParams()  
# fb.setConstraintParams_new()
# capacity_result = fb.runEnv(scenario = "capacity_decrease", mode = 'indoor_dine_in', step = 1)
# #%%
# fb = FB()
# fb.setParams()  
# fb.setConstraintParams_new()
# production_result = fb.runEnv(scenario ="production_decrease", mode = 'indoor_dine_in', step = 1)

#%%
# fb_vis = FB_vis()
# #%%
# emp_analysis = fb_vis.vis(title = "Employee Shortage", result_dict = employee_result)
#  #%%
# supply_analysis = fb_vis.vis(case = "obj_vs_disruption", title = "Supply Shock", result_dict = supply_result)
# #%%
# demand_analysis = fb_vis.vis(title = "Demand Shock", result_dict = demand_result_indoor, mode = "indoor_dine_in")
# #%%
# capacity_analysis = fb_vis.vis(title = "Capacity Shock", result_dict = capacity_result, mode = "indoor_dine_in")
# #%%
# production_analysis = fb_vis.vis(title = "Production Shock", result_dict = production_result, mode = "indoor_dine_in")

# #%%
# emp_analysis_2 = fb_vis.vis(case = "decision_vs_disruption", title = "Employee Shortage", result_dict = employee_result)
# #%%
# supply_analysis_2 = fb_vis.vis(case = "decision_vs_disruption", title = "Supply Shock", result_dict = supply_result)
# #%%
# demand_analysis_2 = fb_vis.vis(case = "decision_vs_disruption", title = "Demand Shock", result_dict = demand_result_indoor, mode = "indoor_dine_in")
# #%%
# capacity_analysis_2 = fb_vis.vis(case = "decision_vs_disruption", title = "Capacity Shock", result_dict = capacity_result, mode = "indoor_dine_in")
# #%%
# production_analysis_2 = fb_vis.vis(case = "decision_vs_disruption", title = "Production Shock", result_dict = production_result, mode = "indoor_dine_in")
# #%%
# # violation analysis
# loc = supply_analysis['system_infeasibility']
# inf_model = supply_result['model'][loc]
# fea_model = supply_result['model'][loc-1]
# constrs_pen = [9,9,9,9,9,9,9,9,10000,17,10000,17,9,0,0,0,0]
# copy_relax = inf_model.copy()
# fb_vis.relax(copy_relax,constrs_pen)
# #%%
# # violation analysis
# loc = emp_analysis['system_infeasibility']
# inf_model = employee_result['model'][loc]
# fea_model = employee_result['model'][loc-1]
# constrs_pen = [9,9,9,9,9,9,9,9,10000,17,10000,17,9,0,0,0,0]
# copy_relax = inf_model.copy()
# fb_vis.relax(copy_relax,constrs_pen)
# #%%
# # test without penalty
# copy_relax = inf_model.copy()
# fb_vis.relax(copy_relax)

# #%%
# # compare across the disruption scenarios
# constrs_pen = [9,9,9,9,9,9,9,9,10000,17,10000,17,9,0,0,0,0]
# prep_dict = {"employee_shortage":{"model": employee_result,"analysis":[emp_analysis, emp_analysis_2]},
#              "supply_shock":{"model": supply_result,"analysis":[supply_analysis, supply_analysis_2]},
#              }
# prep_dict_mode = {"demand_shock":{"mode":{"indoor_dine_in":{"model":demand_result_indoor,
#                                                             "analysis":[demand_analysis,demand_analysis_2]}}},
#                   "capacity_shock":{"mode":{"indoor_dine_in":{"model":capacity_result,
#                                                               "analysis":[capacity_analysis,capacity_analysis_2]}}},
#                   "production_shock":{"mode":{"indoor_dine_in":{"model":production_result,
#                                                                 "analysis":[production_analysis,production_analysis_2]}}}
#                   }
# #%%
# relaxed_result = {}
# for sce in prep_dict.keys():
#     loc = prep_dict[sce]["analysis"][0]["system_infeasibility"]
#     inf_model = prep_dict[sce]['model']['model'][loc]
#     copy_relax = inf_model.copy()
#     copy_IIS = inf_model.copy()
#     relaxed_model = fb_vis.relax(copy_relax, constrs_pen)
#     relaxed_result[sce] = relaxed_model.ObjVal
#     # print(relaxed_model.ObjVal)
#     # IIS_result, removed = fb_vis.check_IIS(copy_IIS)
# #%%
# relaxed_result_mode = {}
# # the modes cou
# modes = ["indoor_dine_in"]
# for sce in prep_dict_mode.keys():
#     for mode in modes:
#         relaxed_result_mode[sce] = {}
#         loc = prep_dict_mode[sce]["mode"][mode]["analysis"][0]["system_infeasibility"]
#         if loc > 100:
#             break
#         inf_model = prep_dict_mode[sce]["mode"][mode]['model']['model'][loc]
#         copy_relax = inf_model.copy()
#         copy_IIS = inf_model.copy()
#         relaxed_model = fb_vis.relax(copy_relax, constrs_pen)
#         relaxed_result_mode[sce][mode] = relaxed_model.ObjVal
#%%

# file = {"demand": {"indoor_dine_in": 31,
#                    "outdoor_dine_in": 30,
#                    "takeout": 30,
#                    "delivery": 16},
#         "default_supply":{"indoor_dine_in": 1.4 * (1/3) * ( 2 + 5.5 + 2.5 + 3+ 6),
#                            "outdoor_dine_in": 1.4 * (1/3) * ( 2 + 5.5 + 2.5 + 3+ 6),
#                            "takeout": 1.4 * (1/3) * ( 2 + 5.5 + 2.5 + 3+ 6),
#                            "delivery": 1.4 * (1/3) * ( 2 + 5.5 + 2.5 + 3+ 6)},
#         "input_costs": {"indoor_dine_in": {"employee": 17* self.working_hr /4,
#                                           "supply": 1 *1.4 * (1/3) * ( 2 * 0.7 + 5.5* 0.3 + 2.5*1 + 3*0.22 + 6*0.09) * self.price_adj},
#                         "outdoor_dine_in":{"employee": 17* self.working_hr / 4,
#                                           "supply": 1 *1.4 * (1/3) * ( 2 * 0.7 + 5.5* 0.3 + 2.5*1 + 3*0.22 + 6*0.09) * self.price_adj},
#                         "takeout": {"employee": 15*self.working_hr / 4,
#                                    "supply": 1 * 1.4 * (1/3) * ( 2 * 0.7 + 5.5* 0.3 + 2.5*1 + 3*0.22 + 6*0.09)* self.price_adj },
#                         "delivery": {"employee": 17*self.working_hr / 4,
#                                     "supply": 1 * 1.4* (1/3) * ( 2 * 0.7 + 5.5* 0.3 + 2.5*1 + 3*0.22 + 6*0.09) * self.price_adj},},
#         "fixed_costs": {"indoor_dine_in": 70 * self.fixed_adj,
#                            "outdoor_dine_in": 70 * self.fixed_adj,
#                            "takeout": 70 * self.fixed_adj,
#                            "delivery": 70 * self.fixed_adj,},
#         "price": {"indoor_dine_in": 30,
#                       "outdoor_dine_in": 30,
#                       "takeout": 30,
#                       "delivery": 45,},
#         "capacity":  {"indoor_dine_in": 31,
#                          "outdoor_dine_in": 30,
#                          "takeout": 30,
#                          "delivery": 16,},
#         "full_quality":  {"indoor_dine_in": 13.948873,
#                          "outdoor_dine_in": 12.509401,
#                          "takeout": 12.509401,
#                          "delivery": 1.586302,},
#         "rent_utility": 116.67,
#         "A": 0.7,
#         "alpha": 0.7,
#         "employee_ub" : 50,
#         "supply_ub" : 100,
#         "prod_ub" : 0.7,
        
#         # disruptions
#         "employee_shortage" : 100, # 
#         "supply_shortage" : 100,
#         "demand_decrease" : {"indoor_dine_in": 100,
#                              "outdoor_dine_in": 100,
#                              "takeout": 100,
#                              "delivery": 100,},
#         "production_decrease" : {"indoor_dine_in": 100,
#                                  "outdoor_dine_in": 100,
#                                  "takeout": 100,
#                                  "delivery": 100,},
#         "capacity_decrease" : {"indoor_dine_in": 100,
#                                "outdoor_dine_in": 100,
#                                "takeout": 100,
#                                "delivery": 100,},
#         "cost_e_increase" : 5.4,
#         "cost_s_increase"  : 8.8,
#         "delta" : {"indoor_dine_in": 1,
#                    "outdoor_dine_in": 0,
#                    "takeout": 1,
#                    "delivery": 1,},
#         }

