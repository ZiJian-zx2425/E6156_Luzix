import { Component } from '@angular/core';
import {Appointment, AppointmentService} from '../services/appointment.service';

@Component({
  selector: 'app-schedule',
  templateUrl: './schedule.component.html',
  styleUrls: ['./schedule.component.css'],
})
export class ScheduleComponent {
  newAppointment = {
    patientName: '',
    doctorName: '',
    dateTime: ''
  };

  appointments: Appointment[] = [];
  // Variable to store feedback message
  message: string = '';

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

  scheduleAppointment() {
    if(this.newAppointment.patientName && this.newAppointment.doctorName && this.newAppointment.dateTime) {
      this.appointmentService.scheduleAppointment(this.newAppointment).subscribe({
        next: (response) => {
          console.log(response);
          this.appointments.push(response);
          this.message = 'Appointment scheduled successfully!';
          this.resetForm();
        },
        error: (err) => {
          console.error(err);
          this.message = 'An error occurred while scheduling the appointment.';
        }
      });
    } else {
      this.message = 'Please fill in all fields to schedule an appointment.';
    }
  }

  resetForm() {
    this.newAppointment = {
      patientName: '',
      doctorName: '',
      dateTime: ''
    };
  }
}
