#Pandas Profile Report
from ydata_profiling import ProfileReport
from comp647 import df

profile = ProfileReport(df,title="Profiling Report")
profile.to_file("ProfilingReport.html")
profile.to_file("ProfilingReport.json")