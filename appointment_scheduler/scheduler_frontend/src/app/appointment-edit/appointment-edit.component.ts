import {Component, OnInit} from '@angular/core';
import {AppointmentService} from "../services/appointment.service";
import {ActivatedRoute, Router} from "@angular/router";

@Component({
  selector: 'app-appointment-edit',
  templateUrl: './appointment-edit.component.html',
  styleUrls: ['./appointment-edit.component.css']
})
export class AppointmentEditComponent implements OnInit {
  appointment: any; // Replace with the correct type
  appointmentId: any;

  constructor(
      private appointmentService: AppointmentService,
      private route: ActivatedRoute,
      private router: Router
  ) {}

  ngOnInit() {
    this.appointmentId = this.route.snapshot.params['id'];
    this.loadAppointment();
  }

  loadAppointment() {
    if (this.appointmentId) {
      this.appointmentService.getAppointmentById(this.appointmentId).subscribe(
          (data) => {
            this.appointment = data;
          },
          (error) => {
            console.error('Error loading appointment', error);
            // Handle not found or other errors here
          }
      );
    } else {
      console.error('Appointment ID is undefined');
      // Handle the error appropriately
    }
  }

  updateAppointment() {
    this.appointmentService.updateAppointment(this.appointmentId, this.appointment).subscribe(
        response => {
          console.log('Appointment updated', response);
          this.router.navigate(['/appointments']); // Navigate back to the list
        },
        error => console.error('Error updating appointment', error)
    );
  }

}
