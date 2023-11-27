import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface Appointment {
    id?: number; // Optional because it's not needed when creating a new appointment
    patientName: string;
    doctorName: string;
    dateTime: string;
}

@Injectable({
    providedIn: 'root'
})
export class AppointmentService {
    private flaskEndpoint = 'http://127.0.0.1:5000/appointments'; // Adjust if needed

    constructor(private http: HttpClient) { }

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



}
