# Nasim .yaml Notes

------------
This .md file is a description of the .yaml environments from pathway:

```
Network_Attack_Simulator\nasim\scenarios\benchmark
```


## subents
the length of the array is the number of subnets. The sum of all the numbers in the subnets array are the number of hosts in the network. 


## topology
The network structure

## sensitive hosts
Hosts that more sensitive and therefore accessing reaps a higher rewards

## os
Operating system 
- linux
- windows

## processes
Background Processes

#### Apache Tomcat:
Open source implementation of Java Servlet, JavaServer, and Websocket tech. "Pure Java" HTTP web server development environment

This service reached end of life on June 39 2018. Meaning any new vulnerabilities after this point have not been patched. 

There are 67 entries for exploits listed on the exploit database. Of those, there are 5 downloadable files in the exploit database for tomcat. 
https://www.exploit-db.com/search?q=tomcat 

#### DACLSVM:
Direscretionary Access Control List, is an internal list attatched to to an object in Active Directory that specifies which users and groups can access the object and what kinds of operations they can preform. 

This can be exploited:
https://medium.com/@orhan_yildirim/windows-privilege-escalation-insecure-service-permissions-e4f33dbff219

As long as we have permissions on the service, we can change the fie executable file loaction on the target system. This is a form of
privilege escalation. 

#### Schtasks:
https://docs.microsoft.com/en-us/previous-versions/windows/it-pro/windows-server-2012-R2-and-2012/cc725744(v=ws.11)

For Windows Server 2008, Windows Server 2012, and Windows 8

Schedules commands and programms to run periodically or at a specific time. 

This can be exploited: 
https://redcanary.com/threat-detection-report/techniques/scheduled-task/

Ranked 4th when counting only sub-techniques for adversaries. 

You can't turn of scheduled tasks, therefore they enable adversaries to inconspicuously conduct an array of malicious activity.

- Establish persistance in an environment
- execute processes, ideally with priveliges and at customized intervals


## exploits

<u>prob</u>: Probability of successful execution
<u>cost</u>: Reward cost to the execute process regardless of it is successful
<u>access</u>: The level of access required to execute the process. 



## host_configuration

This just lists each host on from the topology grid. This defines os, and an array of services and process

## firewall
another part of the topology grid, tells the type of service used as well and its location on the grid. 


## privilege escalation
All the processes have known vulnerabilities and the ability to have an attacker commit privilege escalation on them. 


## scans
all scans have costs too. These are from the base action class. 
- service scan
- os scan
- subnet scan
- process scan

## services
- ssh
- ftp
- http
- smtp
- samba



## my_network.yaml output

the yaml code was put into the "tiny" network as I couldn't find the part in the code to add my_network.yaml to the list of available networks.

```
C:\Users\Shreyas\Documents\GitHub\ESD_NASim_Summer2022\Network_Attack_Simulator\nasim\agents>python bruteforce_agent.py tiny
C:\Users\Shreyas\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\matplotlib\__init__.py:152: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
  if LooseVersion(module.__version__) < minver:
C:\Users\Shreyas\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\setuptools\_distutils\version.py:351: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
  other = LooseVersion(other)
18
------------------------------------------------------------
STARTING EPISODE
------------------------------------------------------------
t: Reward
17: -18.0
35: 64.0
------------------------------------------------------------
EPISODE FINISHED
------------------------------------------------------------
Goal reached = True
Total steps = 47
Total reward = 153.0
```