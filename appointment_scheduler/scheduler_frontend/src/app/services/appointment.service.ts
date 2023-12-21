import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface Appointment {
    status: string;
    id?: number; // Optional because it's not needed when creating a new appointment
    patientName: string;
    doctorName: string;
    dateTime: string;
}

@Injectable({
    providedIn: 'root'
})
export class AppointmentService {
    // private flaskEndpoint = 'http://127.0.0.1:5000/appointments'; // local
    private flaskEndpoint = 'http://127.0.0.1:5001/appointments'; // local
    private officeHoursUrl = 'http://127.0.0.1:5001/officehours'; // URL to your Flask API
    // private flaskEndpoint = 'http://184.73.151.162:5000/appointments'; // EC2

    constructor(private http: HttpClient) { }

    getAvailableSlots(doctorName: string): Observable<string[]> {
        // Fetch available slots for the given doctor name
        return this.http.get<string[]>(`${this.officeHoursUrl}?doctor_name=${encodeURIComponent(doctorName)}`);
    }

    // GET
    getAppointments(): Observable<Appointment[]> {
        return this.http.get<Appointment[]>(this.flaskEndpoint);
    }

    // POST: /appointments/
    scheduleAppointment(appointment: any): Observable<any> {
        //return this.http.post<any>(`${this.flaskEndpoint}/appointments`, newAppointment);
        return this.http.post<Appointment>(this.flaskEndpoint, appointment);
    }

    // DELETE: /appointments/${id}
    cancelAppointment(id: number): Observable<any> {
        return this.http.delete<any>(`${this.flaskEndpoint}/${id}`);
    }

    // PUT (update):  /appointments/${id}
    updateAppointment(id: number, appointment: Appointment): Observable<Appointment> {
        // Ensure that appointment has an id, as it's needed to update the correct appointment
        if (id == null) {
            throw new Error('Appointment ID is required for updating.');
        }
        return this.http.put<Appointment>(`${this.flaskEndpoint}/${id}`, appointment);
    }

    // Method to retrieve an appointment by its ID
    getAppointmentById(id: number): Observable<Appointment> {
        return this.http.get<Appointment>(`${this.flaskEndpoint}/${id}`);
    }

    updateAppointmentStatus(appointmentId: number, status: string): Observable<any> {
        return this.http.put(`${this.flaskEndpoint}/${appointmentId}`, { status });
    }

}
