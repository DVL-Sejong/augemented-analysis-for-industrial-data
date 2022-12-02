import java.sql.Time;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;



public class LifeMapParameter {
	/**
	 * Fixed Variables
	 */
	// Sensor type, bit operation
	public static final int SENSOR_ACCEL = 1;
	public static final int SENSOR_MAG = (1 << 1);
	public static final int SENSOR_GPS = (1 << 2);
	public static final int SENSOR_WIFI = (1 << 3);
	public static final int SENSOR_NETWORK = (1 << 4);
	public static final int SENSOR_TEMPERATURE = (1 << 5);
	public static final int SENSOR_WIFI_STATE = (1 << 6);
	public static final int SENSOR_PREDICTION = (1 << 7);
	public static final int SENSOR_BLUETOOTH = (1 << 8);
	public static final int SENSOR_ACTIVATED = (1 << 30);

	// activity type, bit operation
	public static final int ACTIVITY_STAY = 1;
	public static final int ACTIVITY_MOVE = (1 << 1);
	public static final int ACTIVITY_WALK = (1 << 2);
	public static final int ACTIVITY_RUN = (1 << 3);
	public static final int ACTIVITY_STAY_HOUR = (1 << 4);
	
	public static final double EPS = 0.00000001;
	
	public static final int UNKNOWN_LOCATION = 0;
	public static final int UNKNOWN_ACCURACY = 100000;	// meter, heuristic 

	public static final int MINIMUM_SIGNAL = -100;	// dB

	public static final long SECOND = 1000;
	public static final long MINUTE = SECOND * 60;
	public static final long HOUR = MINUTE * 60;
	public static final long DAY = 24 * HOUR;
	public static final long WEEK = 7 * DAY;
	public static final long MONTH = 30 * DAY;
	public static final long YEAR = 365 * DAY;

	/**
	 * millisecond to day, hour, and minute string
	 * @param l
	 * @return (day)d (hour)h (minute)m
	 */
	public static String longToDHM(long l) {
		String r = "";
		if (l / DAY > 0)
			r += Long.toString(l / DAY) + "d ";
		if (l % DAY / HOUR > 0)
			r += Long.toString(l % DAY / HOUR) + "h ";
		r += Long.toString(l % HOUR / MINUTE) + "m";
		return r;
	}

    /**
     * 
     * @param lat1
     * @param lon1
     * @param lat2
     * @param lon2
     * @return the distance between two points in meters
     */
    public static double getDistance(double lat1, double lon1, double lat2, double lon2) {
    	double theta = lon1 - lon2;
    	double dist = Math.sin(deg2rad(lat1)) * Math.sin(deg2rad(lat2)) + Math.cos(deg2rad(lat1)) * Math.cos(deg2rad(lat2)) * Math.cos(deg2rad(theta));
    	dist = Math.acos(dist);
    	dist = rad2deg(dist);
    	dist = dist * 60 * 1.1515;
    	dist = dist * 1.609344;
		return (dist);
    }
    private static double deg2rad(double deg) {
    	  return (deg * Math.PI / 180.0);
   	}
    private static double rad2deg(double rad) {
    	  return (rad * 180.0 / Math.PI);
   	}
}
