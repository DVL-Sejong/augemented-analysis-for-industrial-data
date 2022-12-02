import java.sql.ResultSet;
import java.sql.SQLException;

/* 
 * Copyright 2009 Chon Yohan, Yonsei University 
 * 
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License. 
 * You may obtain a copy of the License at 
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0 
 * 
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
 * See the License for the specific language governing permissions and 
 * limitations under the License. 
 */

/**
 * Describes the state of Wifi access point.
 * 
 * @author Chon John (john@mobed.yonsei.ac.kr)
 * @since 1.5
 * @version 1.5
 */
public class WifiInfo implements Cloneable, Comparable<WifiInfo> {
	private String BSSID;
	private String SSID;
	private long sampleCount;
	private float signal;
	private float signalDeviation;
	private String open;
	private String time;
	
	private long id;
	private long nodeId;
	private boolean dirty;

	/**
	 * Constructs a new WifiInfo. 
	 * By default, signal, sampleCount are 0; BSSID, SSID, open are null.
	 */
	public WifiInfo() {
		this.clear();
	}

	/**
	 * Clear the contents.
	 */
	public void clear() {
		BSSID = null;
		SSID = null;
		signal = 0;
		signalDeviation = 0;
		sampleCount = 0;
		open = null;
		time = null;
		id = 0;
		nodeId = 0;
		dirty = false;
	}

	/**
	 * Returns sampling count of this access point.
	 */
	public long getSampleCount() {
		return sampleCount;
	}

	/**
	 * Sets sampling count of this access point.
	 */
	public void setSampleCount(long sampleCount) {
		this.sampleCount = sampleCount;
	}

	/**
	 * Returns the basic service set identifier (BSSID) of this access point.
	 * 
	 * @return the BSSID, in the form of a six-byte MAC address:
	 *         XX:XX:XX:XX:XX:XX
	 */
	public String getBSSID() {
		return BSSID;
	}

	/**
	 * Sets the basic service set identifier (BSSID) of this access point.
	 * 
	 * @param bssid
	 *            the BSSID, in the form of a six-byte MAC address:
	 *            XX:XX:XX:XX:XX:XX
	 */
	public void setBSSID(String bssid) {
		BSSID = bssid;
	}

	/**
	 * Returns the service set identifier (SSID) of this access point.
	 */
	public String getSSID() {
		return SSID;
	}

	/**
	 * Sets the service set identifier (SSID) of this access point.
	 */
	public void setSSID(String ssid) {
		ssid.replaceAll(",", ".");
		SSID = ssid;
	}

	/**
	 * Returns the received signal strength indicator of this access point.
	 */
	public float getSignal() {
		return signal;
	}

	/**
	 * Sets the received signal strength indicator of this access point.
	 */
	public void setSignal(float signal) {
		this.signal = signal;
	}

	/**
	 * Returns the authentication, key management, and encryption schemes
	 * supported by the access point.
	 */
	public String getOpen() {
		return open;
	}

	/**
	 * Sets the authentication, key management, and encryption schemes supported
	 * by the access point.
	 */
	public void setOpen(String open) {
		this.open = open;
	}

	
	public long getId() {
		return id;
	}

	public void setId(long id) {
		this.id = id;
	}

	public boolean isDirty() {
		return dirty;
	}

	public void setDirty(boolean dirty) {
		this.dirty = dirty;
	}

	public float getSignalDeviation() {
		return signalDeviation;
	}

	public void setSignalDeviation(float signalDeviation) {
		this.signalDeviation = signalDeviation;
	}

	public String getTime() {
		return time;
	}
	public void setTime(String time) {
		this.time = time;
	}
	
	public long getNodeId() {
		return nodeId;
	}

	public void setNodeId(long nodeId) {
		this.nodeId = nodeId;
	}

	public String toString() {
		String s = "";
		
		s += id + " " + SSID + "(" + BSSID + ") " + String.format("%.1f", signal) + "/" + String.format("%.1f", signalDeviation) + " of " + sampleCount + " at " + time + "\n";
		
		return s;
	}
	public void merge(WifiInfo w) {
		float wSignal = w.getSignal();
		long wCount = w.getSampleCount();
		float wDeviation = w.getSignalDeviation();
		float mSignal = (wSignal * wCount + signal * sampleCount) / (wCount + sampleCount);
		float mSignalSquare = wCount * (wDeviation * wDeviation + wSignal * wSignal);
		float _mSignalSquare = sampleCount * (signalDeviation * signalDeviation + signal * signal);
		float mSignalDeviation = (float)Math.sqrt((mSignalSquare + _mSignalSquare) / (wCount + sampleCount) - mSignal * mSignal);

		signal = mSignal;
		signalDeviation = mSignalDeviation;
		sampleCount += wCount;
		
		if(time == null && w.getTime() != null)
			time = w.getTime();
		else if(time != null && w.getTime() != null && time.compareTo(w.getTime()) < 0)
			time = w.getTime();
	}
	public Object clone() {
		WifiInfo o = null;
		try {
			o = (WifiInfo)super.clone();
			o.BSSID = BSSID;
			o.SSID = SSID;
			o.sampleCount = sampleCount;
			o.signal = signal;
			o.signalDeviation = signalDeviation;
			o.open = open;
			o.time = time;
			o.id = id;
			o.dirty = dirty;
		} catch (CloneNotSupportedException e) {
			e.printStackTrace();
		}

		return o;
	}
	@Override
	public boolean equals(Object obj) {
		return this.BSSID.equals(((WifiInfo)obj).BSSID);
	}
	public void set(ResultSet cs) throws SQLException {
		id = cs.getLong(cs.findColumn(LifeMapDatabase.AP_KEY_ID));
		nodeId = cs.getLong(cs.findColumn(LifeMapDatabase.AP_KEY_NODE_ID));
		BSSID = cs.getString(cs.findColumn(LifeMapDatabase.AP_KEY_BSSID));
		if(cs.getString(cs.findColumn(LifeMapDatabase.AP_KEY_SSID)) != null)
			SSID = cs.getString(cs.findColumn(LifeMapDatabase.AP_KEY_SSID));
		if(cs.getString(cs.findColumn(LifeMapDatabase.AP_KEY_OPEN)) != null)
			open = cs.getString(cs.findColumn(LifeMapDatabase.AP_KEY_OPEN));
		signal = cs.getFloat(cs.findColumn(LifeMapDatabase.AP_KEY_SIGNAL));
		signalDeviation = cs.getFloat(cs.findColumn(LifeMapDatabase.AP_KEY_SIGNAL_DEVIATION));
		sampleCount = cs.getInt(cs.findColumn(LifeMapDatabase.AP_KEY_SAMPLE_COUNT));
		time = cs.getString(cs.findColumn(LifeMapDatabase.AP_KEY_TIME));
	}

	@Override
	public int compareTo(WifiInfo another) {
		return this.BSSID.compareTo(another.BSSID);
	}
}
