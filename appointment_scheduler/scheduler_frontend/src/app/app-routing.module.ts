import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import {ScheduleComponent} from "./schedule/schedule.component";
import {AppointmentEditComponent} from "./appointment-edit/appointment-edit.component";
import { AppointmentListComponent } from './appointment-list/appointment-list.component';


const routes: Routes = [
  { path: '', redirectTo: '/schedule', pathMatch: 'full' },
  { path: 'schedule', component: ScheduleComponent },
  { path: 'appointments', component: AppointmentListComponent },
  { path: 'edit-appointment/:id', component: AppointmentEditComponent }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
