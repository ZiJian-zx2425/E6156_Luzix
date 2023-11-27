// In appointment-list.component.ts
import { Component, OnInit } from '@angular/core';
import { AppointmentService, Appointment } from '../services/appointment.service';

@Component({
  selector: 'app-appointment-list',
  templateUrl: './appointment-list.component.html',
  styleUrls: ['./appointment-list.component.css']
})
export class AppointmentListComponent implements OnInit {
  appointments: Appointment[] = [];
  constructor(private appointmentService: AppointmentService) {}
  ngOnInit(): void {
    this.loadAppointments();
  }

  loadAppointments(): void {
    this.appointmentService.getAppointments().subscribe(
        (data) => {
          this.appointments = data;
        },
        (error) => {
          console.error('Error fetching appointments', error);
          // Handle the error
        }
    );
  }

  cancelAppointment(id: any): void {
    if (confirm('Are you sure you want to delete this appointment?')) {
      this.appointmentService.cancelAppointment(id).subscribe(() => {
        // After deletion, refresh the list or remove the item from the local array
        console.log('Appointment deleted');
        this.appointments = this.appointments.filter(appointment => appointment.id !== id);
      });
    }
  }

}
