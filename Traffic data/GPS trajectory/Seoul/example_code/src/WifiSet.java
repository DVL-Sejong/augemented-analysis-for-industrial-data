

import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Collections;
import java.util.Vector;

public class WifiSet implements Cloneable {
	Vector<WifiInfo> v;
	
	public WifiSet() {
		v = new Vector<WifiInfo>();
	}
	public void clear() {
		v.clear();
	}
	public int size() {
		return v.size();
	}
	public WifiInfo get(int index) {
		return v.get(index);
	}
	public void add(WifiSet ws) {
		for(int i=0 ; i < ws.size() ; i++) {
			this.add(ws.get(i));
		}
	}
	public void add(WifiInfo wi) {
		boolean find = false;
		for (int i=0 ; i < v.size() ; i++) {
			WifiInfo w = v.elementAt(i);
			if (w.equals(wi)) {
				find = true;
				float wSignal = w.getSignal();		float _wSignal = wi.getSignal();
				long wCount = w.getSampleCount();	long _wCount = wi.getSampleCount();
				float wDeviation = w.getSignalDeviation();	float _wDeviation = wi.getSignalDeviation();
				float signal = (wSignal * wCount + _wSignal * _wCount) / (wCount + _wCount);
				float wSignalSquare = wCount * (wDeviation * wDeviation + wSignal * wSignal);
				float _wSignalSquare = _wCount * (_wDeviation * _wDeviation + _wSignal * _wSignal);
				float signalDeviation = (float)Math.sqrt((wSignalSquare + _wSignalSquare) / (wCount + _wCount) - signal * signal);
				
				w.setSignal(signal);
				w.setSignalDeviation(signalDeviation);
				w.setSampleCount(wCount + _wCount);
				if(w.getTime() == null && wi.getTime() != null)
					w.setTime(wi.getTime());
				else if(w.getTime() != null && wi.getTime() != null && w.getTime().compareTo(wi.getTime()) > 0)
					w.setTime(wi.getTime());
				w.setDirty(true);
				break;
			}
		}
		if (!find) {
			v.add(wi);
		}
		return;
	}
	public void sort() {
		Collections.sort(v);
	}
	
	public void set(ResultSet cs) throws SQLException {
		while(cs.next()) {
			WifiInfo wi = new WifiInfo();
			wi.setId(cs.getLong(cs.findColumn(LifeMapDatabase.AP_KEY_ID)));
			wi.setBSSID(cs.getString(cs.findColumn(LifeMapDatabase.AP_KEY_BSSID)));
			wi.setSSID(cs.getString(cs.findColumn(LifeMapDatabase.AP_KEY_SSID)));
			wi.setSignal(cs.getFloat(cs.findColumn(LifeMapDatabase.AP_KEY_SIGNAL)));
			wi.setSignalDeviation(cs.getFloat(cs.findColumn(LifeMapDatabase.AP_KEY_SIGNAL_DEVIATION)));
			wi.setOpen(cs.getString(cs.findColumn(LifeMapDatabase.AP_KEY_OPEN)));
			wi.setSampleCount(cs.getLong(cs.findColumn(LifeMapDatabase.AP_KEY_SAMPLE_COUNT)));
			wi.setTime(cs.getString(cs.findColumn(LifeMapDatabase.AP_KEY_TIME)));
			this.add(wi);
		}
	}
	public Object clone() {
		WifiSet o = null;
		try {
			o = (WifiSet)super.clone();
			o.v = (Vector<WifiInfo>)v.clone();
		} catch (CloneNotSupportedException e) {
			e.printStackTrace();
		}

		return o;
	}
}
