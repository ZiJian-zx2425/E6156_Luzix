import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import {ScheduleComponent} from "./schedule/schedule.component";
import {AppointmentEditComponent} from "./appointment-edit/appointment-edit.component";
import { AppointmentListComponent } from './appointment-list/appointment-list.component';
import { AuthGuard } from './services/auth/auth.guard';
import {AuthSuccessComponent} from "./auth-success/auth-success.component";
import {RoleGuard} from "./role.guard";
import {OfficehourComponent} from "./officehour/officehour.component";
import {PostedOfficeHoursComponent} from "./postedofficehours/postedofficehours.component";
import {PatientsRecordsComponent} from "./patients-records/patients-records.component";
import {SearchByPatientComponent} from "./search-by-patient/search-by-patient.component";


const routes: Routes = [
  { path: '', redirectTo: '/schedule', pathMatch: 'full' },
  { path: 'schedule', component: ScheduleComponent, canActivate: [AuthGuard,RoleGuard]},
  { path: 'appointments', component: AppointmentListComponent, canActivate: [AuthGuard]},
  { path: 'edit-appointment/:id', component: AppointmentEditComponent, canActivate: [AuthGuard]},
  { path: 'auth-success', component: AuthSuccessComponent },
  { path: 'officehour', component: OfficehourComponent, canActivate: [AuthGuard]},
  { path: 'postedofficehours', component: PostedOfficeHoursComponent, canActivate: [AuthGuard]},
  { path: 'patients-records', component: PatientsRecordsComponent, canActivate: [AuthGuard] },
  { path: 'search-by-patient', component: SearchByPatientComponent, canActivate: [AuthGuard] }
  /*
  { path: 'schedule', component: ScheduleComponent},
  { path: 'appointments', component: AppointmentListComponent},
  { path: 'edit-appointment/:id', component: AppointmentEditComponent},
  { path: 'auth-success', component: AuthSuccessComponent },
  { path: 'officehour', component: OfficehourComponent},
  { path: 'postedofficehours', component: PostedOfficeHoursComponent},
  { path: 'patients-records', component: PatientsRecordsComponent},
  { path: 'search-by-patient', component: SearchByPatientComponent}

   */
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})

export class AppRoutingModule { }
